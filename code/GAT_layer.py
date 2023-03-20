import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet.ndarray.gen_op import Activation
import numpy as np
import math

class GATLayer(nn.Block):
    def __init__(self, G, embedding_size, seq_hiddim, dropout, slope, ctx):
        super(GATLayer, self).__init__()

        self.TF_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.tg_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.G = G
        self.slope = slope
        self.seq_fc_1 = nn.Dense(seq_hiddim, use_bias=False)
        self.seq_fc_2 = nn.Dense(seq_hiddim, use_bias=False)
        self.TF_fc = nn.Dense(embedding_size, use_bias=False, weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        self.tg_fc = nn.Dense(embedding_size, use_bias=False, weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Dense(1, use_bias=False, weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        self.leaky_relu = nn.LeakyReLU(slope)

    def edge_attention(self, edges):
        z2 = nd.concat(edges.src['z'], edges.dst['z'], dim=1)
        a = self.attn_fc(z2)
        # return {'e': a}
        # a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': self.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = nd.softmax(nodes.mailbox['e'], axis=1)
        h = nd.sum(alpha * nodes.mailbox['z'], axis=1)

        # return {'h': F.elu(h)}
        return {'h': h}

    def forward(self, G):
        G.apply_nodes(lambda nodes: {
                'features': nd.concat(self.seq_fc_1(nodes.data['seq_features']), nodes.data['pg_features'], dim=1)},
                          self.TF_nodes)
        G.apply_nodes(lambda nodes: {
                'features': nd.concat(self.seq_fc_2(nodes.data['seq_features']), nodes.data['pg_features'], dim=1)},
                          self.tg_nodes)
        G.apply_nodes(lambda nodes: {'z': self.dropout(self.TF_fc(nodes.data['features']))}, self.TF_nodes)
        G.apply_nodes(lambda nodes: {'z': self.dropout(self.tg_fc(nodes.data['features']))}, self.tg_nodes)

        G.apply_edges(self.edge_attention)
        G.update_all(self.message_func, self.reduce_func)

        return G.ndata.pop('h')


class GAT(nn.Block):
    def __init__(self, G, embedding_size, seq_hiddim, dropout, slope, ctx):
        super(GAT, self).__init__()

        self.G = G
        self.dropout = dropout
        self.slope = slope

        self.layers = nn.Sequential()
        self.layers.add(GATLayer(G, embedding_size, seq_hiddim, dropout, slope, ctx))

    def forward(self, G):
        for layer in self.layers:
            outs = layer(G)
        return outs
