import numpy as np
import pandas as pd
import dgl
import mxnet as mx
from mxnet import ndarray as nd
from sklearn.utils import shuffle


def load_data(dir, k, species):
    # associations as feature
    df1 = pd.read_csv(dir+species+'/TF_embedding.csv', header=None)
    TF_1 = df1.to_numpy()
    df2 = pd.read_csv(dir+species+'/tg_embedding.csv', header=None)
    tg_1 = df2.to_numpy()

    # sequence as feature
    TF_2 = np.load(dir+species+'/TF_' + k + 'mers.npy')
    tg_2 = np.load(dir+species+'/tg_' + k + 'mers.npy')
    print(species)
    print(k + 'mer')

    return TF_1, tg_1, TF_2, tg_2


def sample(dir, species, random_seed):
    all_associations = pd.read_csv(dir+species+'/tg_TF_pairs.csv', names=['TF', 'tg', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]

    num=known_associations.shape[0]
    sam_positive=known_associations.sample(n=num, random_state=random_seed, axis=0)
    random_negative = unknown_associations.sample(n=num*2, random_state=random_seed, axis=0)

    sample_df = sam_positive.append(random_negative)
    sample_df = shuffle(sample_df)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


def build_graph(dir, k, species, random_seed, ctx):
    # dgl.load_backend('mxnet')
    TF_1, tg_1, TF_2, tg_2 = load_data(dir, k, species)
    TF_1 = np.ascontiguousarray(TF_1)
    tg_1 = np.ascontiguousarray(tg_1)
    TF_2 = np.ascontiguousarray(TF_2)
    tg_2 = np.ascontiguousarray(tg_2)

    samples = sample(dir, species, random_seed)

    print('Building graph ...')
    g1 = dgl.DGLGraph(multigraph=True)
    g1.add_nodes(TF_1.shape[0] + tg_1.shape[0])
    node_type = nd.zeros(g1.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:TF_1.shape[0]] = 1
    g = g1.to(ctx)
    g.ndata['type'] = node_type

    # concate features
    print('Adding features ...')
    seq_data = nd.zeros(shape=(g.number_of_nodes(), TF_2.shape[1]), dtype='float32', ctx=ctx)
    pg_data = nd.zeros(shape=(g.number_of_nodes(), TF_1.shape[1]), dtype='float32', ctx=ctx)
    pg_data[:TF_1.shape[0], :TF_1.shape[1]] = nd.from_numpy(TF_1)
    seq_data[:TF_1.shape[0], :TF_2.shape[1]] = nd.from_numpy(TF_2)
    pg_data[TF_1.shape[0]: TF_1.shape[0] + tg_1.shape[0], :tg_1.shape[1]] = nd.from_numpy(tg_1)
    seq_data[TF_1.shape[0]: TF_1.shape[0] + tg_1.shape[0], :tg_2.shape[1]] = nd.from_numpy(tg_2)
    g.ndata['seq_features'] = seq_data
    g.ndata['pg_features'] = pg_data


    print('Adding edges ...')
    TF_ids = list(range(1, TF_1.shape[0] + 1))
    tg_ids = list(range(1, tg_1.shape[0] + 1))

    TF_ids_invmap = {id_: i for i, id_ in enumerate(TF_ids)}
    tg_ids_invmap = {id_: i for i, id_ in enumerate(tg_ids)}

    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_tg_vertices = [tg_ids_invmap[id_] + TF_1.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_TF_vertices, sample_tg_vertices,
                data={'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_tg_vertices, sample_TF_vertices,
                data={'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})

    g.readonly()
    print('Successfully build graph !!')

    return g, TF_ids_invmap, tg_ids_invmap, TF_1