import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
from mxnet.gluon.nn import activations
from mxnet.ndarray.gen_op import Activation
import numpy as np
from dgl.nn.mxnet import APPNPConv, SAGEConv, GraphConv
from GAT_layer import GAT

class PPRTGI(nn.Block):
    def __init__(self, encoder, decoder):
        super(PPRTGI, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, seq_hiddim, n_layers, G, aggregator, dropout, slope, alpha, ctx):
        super(GraphEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.seq_hiddim = seq_hiddim
        self.n_layers = n_layers
        self.n = n_layers
        self.alpha = alpha
        self.G = G
        self.aggregator = aggregator
        self.dropout = dropout
        self.ctx = ctx
        self.TF_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.tg_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)
        self.layers = nn.Sequential()
        if aggregator == 'GATConv':
            self.Dropout = nn.Dropout(dropout)
            if self.n_layers:
                self.layers.add(GAT(G, self.embedding_size, self.seq_hiddim, dropout, slope, ctx))
            else:
                raise NotImplementedError
        else:
            self.seq_fc_1 = nn.Dense(self.seq_hiddim, use_bias=False)
            self.seq_fc_2 = nn.Dense(self.seq_hiddim, use_bias=False)
            self.TF_emb = Embedding(self.embedding_size, dropout)
            self.tg_emb = Embedding(self.embedding_size, dropout)

            if aggregator == 'APPNPConv':
                self.layers.add(APPNPConv(k=self.n, alpha=self.alpha, edge_drop=dropout))  # The number of iterations K
                self.Dropout = nn.Dropout(dropout)
            elif aggregator == 'SAGEConv':
                self.Dropout = nn.Dropout(dropout)
                if self.n_layers == 1:
                    self.layers.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                             aggregator_type='pool', feat_drop=self.dropout, bias=True,
                                             norm=None, activation=None))
                elif self.n_layers == 2:
                    self.layers.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                             aggregator_type='pool', feat_drop=self.dropout, bias=True,
                                             norm=None, activation=None))
                    self.layers.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                             aggregator_type='pool', feat_drop=self.dropout, bias=True,
                                             norm=None, activation=None))
                else:
                    raise NotImplementedError
            elif aggregator == 'GraphConv':
                self.Dropout = nn.Dropout(dropout)
                if self.n_layers == 1:
                    self.layers.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                              norm='both', weight=True, bias=True, activation=None,
                                              allow_zero_in_degree=True))
                elif self.n_layers == 2:
                    self.layers.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                              norm='both', weight=True, bias=True, activation=None,
                                              allow_zero_in_degree=True))
                    self.layers.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                              norm='both', weight=True, bias=True, activation=None,
                                              allow_zero_in_degree=True))
                else:
                    raise NotImplementedError

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        if self.aggregator == 'GATConv':
            for layer in self.layers:
                feat = layer(G)
                feat = self.Dropout(feat)
        else:
            G.apply_nodes(lambda nodes: {
                'features': nd.concat(self.seq_fc_1(nodes.data['seq_features']), nodes.data['pg_features'], dim=1)},
                          self.TF_nodes)
            G.apply_nodes(lambda nodes: {
                'features': nd.concat(self.seq_fc_2(nodes.data['seq_features']), nodes.data['pg_features'], dim=1)},
                          self.tg_nodes)
            if self.aggregator == 'APPNPConv':
                G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)
                G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)
                feat = G.ndata['h']
                for layer in self.layers:
                    feat = layer(G, feat)
            elif self.aggregator == 'SAGEConv':
                G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)
                G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)
                feat = G.ndata['h']
                for i, layer in enumerate(self.layers):
                    feat = layer(G, feat)
                    if i != len(self.layers) - 1:
                        feat = nd.relu(feat)
                        feat = self.Dropout(feat)
            elif self.aggregator == 'GraphConv':
                G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)
                G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)
                feat = G.ndata['h']
                for i, layer in enumerate(self.layers):
                    feat = layer(G, feat)
                    if i != len(self.layers) - 1:
                        feat = nd.relu(feat)
                        feat = self.Dropout(feat)
        return feat

class Embedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(Embedding, self).__init__()

        self.layer = nn.Sequential()
        with self.layer.name_scope():
            self.layer.add(nn.Dense(embedding_size, use_bias=False))
            self.layer.add(nn.Dropout(dropout))
            

    def forward(self, ndata):
        embedding = self.layer(ndata['features'])

        return embedding



class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_TF, h_tg):
        results_mask = self.activation((nd.dot(h_TF, self.W.data()) * h_tg).sum(1))

        return results_mask
