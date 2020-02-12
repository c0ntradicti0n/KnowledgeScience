import logging
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import pandas as pd
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx, HolE, TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model, restore_model
import os
import tensorflow as tf
import random
from numpy import cumsum
from more_itertools import flatten
from sklearn.utils import Memory
import pprint
from tspy import TSP
import numpy as np
from pandas import CategoricalDtype
from scipy.spatial.distance import cdist
import pprint
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
logging.getLogger().setLevel(logging.INFO)

parser = ArgumentParser(description='Projecting graph to 3d (and embeddings)')
parser.add_argument('csv',
                    nargs='?',
                    type=str,
                    help='csv with n1, n2, rel columns',
                    default="./test")
args = parser.parse_args()

# getting whole wordnet graph
ke_model_path = "./knowledge_graph_model/csv_ke.amplimodel"
ke_wnkeys_path = "./knowledge_graph_model/csv_ke.wnkeys"

table = pd.read_csv(args.csv, sep='|', header=0)
whole_graph = list(zip(table['n1'], table['rel'], table['n2']))


def percentage_split(seq, percentage_dict):
    cdf = cumsum(list(percentage_dict.values()))
    assert cdf[-1] == 1.0
    stops = list(map(int, cdf * len(seq)))
    return {key: seq[a:b] for a, b, key in zip([0] + stops, stops, percentage_dict.keys())}

if True: #not os.path.isfile(ke_wnkeys_path) or not os.path.isfile(ke_model_path):
    pprint.pprint (whole_graph[:60])
    random.shuffle(whole_graph)

    corpus_split_layout = {
        'train': 0.8,
        'test': 0.1,
        'valid': 0.1
    }
    X = percentage_split(whole_graph, corpus_split_layout)
    known_entities = set (flatten([r[0], r[2]] for r in X['train']))

    id2tok = {i:tok for i, tok in enumerate(known_entities)}
    tok2id = {tok:i for i, tok in enumerate(known_entities)}

    import pickle
    with open(ke_wnkeys_path, 'wb') as handle:
        pickle.dump((tok2id, id2tok), handle)

    X['train'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['train']
                           if r[0] in known_entities and r[2] in known_entities])
    X['valid'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['valid']
                           if r[0] in known_entities and r[2] in known_entities])
    X['test'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['test']
                           if r[0] in known_entities and r[2] in known_entities])
    X_train, X_valid = X['train'], X['valid']
    print('Train set size: ', X_train.shape)
    print('Test set size: ', X_valid.shape)
    ke_kwargs = {
        "verbose":True,
        "k":70,
        "epochs":250
    }

    # ComplEx brings double dimensions because of the twofold nature of complex numbers
    model = ComplEx(**ke_kwargs)
    print ("Training...")
    model.fit(X_train)
    save_model(model, model_name_path=ke_model_path)
    # If we don't transpose the multidimensionality of the embeddings to 3D but take just 3-D-embeddings,
    # This can't be with ComplEX because, it will be an even number and 3 is not
    ke_kwargs['k'] = 3
    model2 = TransE(**ke_kwargs)
    model2.fit(X_train)
    save_model(model2, model_name_path=ke_model_path + '2')
else:
    model = restore_model(model_name_path=ke_model_path)
    model2 = restore_model(model_name_path=ke_model_path+'2')
    with open(ke_wnkeys_path, 'rb') as handle:
        tok2id, id2tok = pickle.load(handle)

def find_in_tok2id(w):
    for s in tok2id.keys():
        if w in s:
            print (w, s, "it is alphabetically there")

tok2id = OrderedDict (tok2id)

print("Extracting Embeddings..")
alle = table['n1'].tolist() + table['n2'].tolist()
embedding_map = dict([(str(a), (model.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)]))
                      for a in alle if str(a) in tok2id])
embedding_map2 = dict([(str(a), (model2.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)]))
                      for a in alle if str(a) in tok2id])

embeddings_array = np.array([i[0] for i in embedding_map.values()])
print ("PCA")
embeddings_3d_pca = PCA(n_components=3).fit_transform(embeddings_array)
print ("TSNE")
embeddings_3d_tsne = TSNE(n_components=3).fit_transform(embeddings_array)
print("k2")
embeddings_k2 = np.array([i[0] for i in embedding_map2.values()])

# Check if second dimension is 3
print (embeddings_3d_pca.shape)
print (embeddings_k2.shape)
assert (embeddings_3d_pca.shape[1] == embeddings_k2.shape and embeddings_k2.shape == 3)

print ("pandas")
table = pd.DataFrame(data={'name':list(s.replace("Synset('", '').replace("')", "")
                                                    for s in embedding_map.keys()),
                           'id': [i[1] for i in embedding_map.values()],
                           'x_pca': embeddings_3d_pca[:, 0],
                           'y_pca': embeddings_3d_pca[:, 1],
                           'z_pca': embeddings_3d_pca[:, 2],
                           'x_tsne': embeddings_3d_tsne[:, 0],
                           'y_tsne': embeddings_3d_tsne[:, 1],
                           'z_tsne': embeddings_3d_tsne[:, 2],
                           'x_k2': embeddings_k2[:, 0],
                           'y_k2': embeddings_k2[:, 1],
                           'z_k2': embeddings_k2[:, 2]
                           })

print ('clusters')
import hdbscan
std_args = {
    'algorithm':'best',
    'alpha':1.0,
    'approx_min_span_tree':True,
    'gen_min_span_tree':False,
    'leaf_size':20,
    'memory': Memory(cachedir=None),
    'metric':'euclidean',
    'min_cluster_size':13,
    'min_samples':None,
    'p':None
}

def cluster(embeddings_array, **kwargs):
    print ('dimensionality', embeddings_array.shape)
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(np.array(embeddings_array))
    print ('number of clusters: ', max(clusterer.labels_))
    return clusterer.labels_

table['cl_pca'] =  cluster(embeddings_3d_pca, **std_args)
table['cl_tsne'] = cluster(embeddings_3d_tsne, **std_args)
table['cl_k2'] =   cluster(embeddings_k2, **std_args)
table['cl_kn'] =   cluster(embeddings_array, **std_args)
table.to_csv("./knowledge_graph_coords/knowledge_graph_3d_choords.csv",
             sep='\t',
             header=True,
             index=False)
table = pd.read_csv("./knowledge_graph_coords/knowledge_graph_3d_choords.csv",
             index_col=0,
             sep='\t')
things = ['pca', 'tsne', 'k2', 'kn']

def make_path (X, D):
    tsp = TSP()

    # Using the data matrix
    tsp.read_data(X)

    # Using the distance matrix
    tsp.read_mat(D)

    from tspy.solvers import TwoOpt_solver

    TwoOpt_solver(initial_tour='NN', iter_num=100000)
    best_tour = tsp.get_best_solution()

    #tsp.plot_solution('TwoOpt_solver')
    return best_tour



for kind in things:
    print ("writing table for %s " % kind)
    table['cl'] = table['cl_%s' % kind]
    cl_cols = table[['cl_%s' % k for k in things]]
    cl_df = table.groupby(by='cl').mean().reset_index()

    # Initialize fitness function object using coords_list
    print ("optimizing the path through all centers")
    if kind == "kn":
        subkind = "tsne"
    else:
        sub_kind = kind

    subset = cl_df[[c + "_" + sub_kind for c in ['x', 'y', 'z']]]
    print (subset[:10])

    points = [list(x) for x in subset.to_numpy()]
    print (points[:10])
    print (len(points))

    arr = np.array(points)
    dist = Y = cdist(arr, arr, 'euclidean')
    new_path = make_path(np.array(points), dist)[:-1]
    print (new_path)

    cl_df[['cl_%s' % k for k in things]] = cl_cols

    path_order_categories = CategoricalDtype(categories=new_path,  ordered = True)
    cl_df['cl_%s' % kind] = cl_df['cl'].astype(path_order_categories)
    cl_df.sort_values(['cl_%s' % kind], inplace=True)
    cl_df['cl_%s' % kind] = cl_df['cl'].astype('int32')
    cl_df.to_csv(
        f'./knowledge_graph_coords/{kind}_clusters_mean_points.csv',
        sep='\t',
        header=True,
        index=False)
    print (kind + " " + str(new_path))

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zip_path = 'data.zip'
zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
zipdir('knowledge_graph_coords', zipf)
zipf.close()

logging.info(f"ampligraph and clustering finished and data written to {zip_path}")