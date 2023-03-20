import time
import random
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics
from utils import build_graph, sample
from model import PPRTGI, GraphEncoder, BilinearDecoder

import dgl.function as FN
import sys
np.set_printoptions(threshold=sys.maxsize)

def Train(dir, k, species, epochs, aggregator, embedding_size, seq_hiddim, layers, dropout, slope, alpha, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    g, TF_ids_invmap, tg_ids_invmap, TFdata = build_graph(dir, k, species, random_seed=random_seed, ctx=ctx)

    samples = sample(dir, species, random_seed=random_seed)

    samples_df = pd.DataFrame(samples, columns=['TF', 'tg', 'label'])
    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_tg_vertices = [tg_ids_invmap[id_] + TFdata.shape[0] for id_ in samples[:, 1]]

    num = len(samples_df)
    index = np.arange(num)
    test_idx = np.random.choice(index, size=int(0.1 * num), replace=False)
    idx = np.delete(index, test_idx, axis=0)
    val_idx = np.random.choice(idx, size=int(0.1 * num), replace=False)
    train_idx = np.setdiff1d(idx, val_idx)

    samples_df['train'] = 0
    samples_df['test'] = 0
    samples_df['val'] = 0

    samples_df['train'].iloc[train_idx] = 1
    samples_df['test'].iloc[test_idx] = 1
    samples_df['val'].iloc[val_idx] = 1

    train_tensor = nd.from_numpy(samples_df['train'].values.astype('int32')).copyto(ctx)
    test_tensor = nd.from_numpy(samples_df['test'].values.astype('int32')).copyto(ctx)
    val_tensor = nd.from_numpy(samples_df['val'].values.astype('int32')).copyto(ctx)

    edge_data = {'train': train_tensor,
                 'test': test_tensor,
                 'val': val_tensor,}

    g.edges[sample_TF_vertices, sample_tg_vertices].data.update(edge_data)
    g.edges[sample_tg_vertices, sample_TF_vertices].data.update(edge_data)


    train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
    g_train = g.edge_subgraph(train_eid, preserve_nodes=True)

    # get the training set
    rating_train = g_train.edata['rating']
    src_train, dst_train = g_train.all_edges()

    # get the validating edge set
    val_eid = g.filter_edges(lambda edges: edges.data['val']).astype('int64')
    src_val, dst_val = g.find_edges(val_eid)
    rating_val = g.edges[val_eid].data['rating']

    # get the testing edge set
    test_eid = g.filter_edges(lambda edges: edges.data['test']).astype('int64')
    src_test, dst_test = g.find_edges(test_eid)
    rating_test = g.edges[test_eid].data['rating']


    src_train = src_train.copyto(ctx)
    src_test = src_test.copyto(ctx)
    src_val= src_val.copyto(ctx)
    dst_train = dst_train.copyto(ctx)
    dst_test = dst_test.copyto(ctx)
    dst_val = dst_val.copyto(ctx)
    print('## Training edges:', len(train_eid))
    print('## Testing edges:', len(test_eid))
    print('## Validating edges:', len(val_eid))

    # Train the model
    model = PPRTGI(GraphEncoder(embedding_size=embedding_size, seq_hiddim=seq_hiddim, n_layers=layers, G=g_train, aggregator=aggregator,
                                  dropout=dropout, slope=slope, alpha=alpha, ctx=ctx),
                     BilinearDecoder(feature_size=embedding_size))

    model.initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
    loss_fn = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    #early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    val_losses = []
    print(model)

    for epoch in range(epochs):
        start = time.time()
        with mx.autograd.record():
            # score_train,embeddings = model(g_train, src_train, dst_train)
            embeddings = model.encoder(g_train)
            score_train = model.decoder(embeddings[src_train], embeddings[dst_train])
            loss_train = loss_fn(score_train, rating_train).mean()
            loss_train.backward()
        trainer.step(1, ignore_stale_grad=True)
        nd.waitall()

        h_val = model.encoder(g)
        score_val = model.decoder(h_val[src_val], h_val[dst_val])
        nd.waitall()

        loss_val = loss_fn(score_val, rating_val).mean()

        train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(score_train.asnumpy()))
        train_aupr = metrics.average_precision_score(np.squeeze(rating_train.asnumpy()),
                                                     np.squeeze(score_train.asnumpy()))
        val_auc = metrics.roc_auc_score(np.squeeze(rating_val.asnumpy()), np.squeeze(score_val.asnumpy()))
        val_aupr = metrics.average_precision_score(np.squeeze(rating_val.asnumpy()),
                                                   np.squeeze(score_val.asnumpy()))

        results_val = [0 if j < 0.5 else 1 for j in np.squeeze(score_val.asnumpy())]
        accuracy_val = metrics.accuracy_score(rating_val.asnumpy(), results_val)
        precision_val = metrics.precision_score(rating_val.asnumpy(), results_val)
        recall_val = metrics.recall_score(rating_val.asnumpy(), results_val)
        f1_val = metrics.f1_score(rating_val.asnumpy(), results_val)


        end = time.time()

        train_losses.append(loss_train.asscalar())
        val_losses.append(loss_val.asscalar())

        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),
              'Val Loss: %.4f' % loss_val.asscalar(),
              'Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
              'F1: %.4f' % f1_val, 'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
              'Train AUPR: %.4f' % train_aupr, 'Val AUPR: %.4f' % val_aupr,
              'Time: %.2f' % (end - start))

    h_test = model.encoder(g)
    score_test = model.decoder(h_test[src_test], h_test[dst_test])

    fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(rating_test.asnumpy()), np.squeeze(score_test.asnumpy()))
    test_auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(np.squeeze(rating_test.asnumpy()),
                                                                   np.squeeze(score_test.asnumpy()))
    test_aupr = metrics.auc(recall, precision)

    results_test = [0 if j < 0.5 else 1 for j in np.squeeze(score_test.asnumpy())]

    accuracy_test = metrics.accuracy_score(rating_test.asnumpy(), results_test)
    precision_test = metrics.precision_score(rating_test.asnumpy(), results_test)
    recall_test = metrics.recall_score(rating_test.asnumpy(), results_test)
    f1_test = metrics.f1_score(rating_test.asnumpy(), results_test)


    print('Test AUC: %.4f' % test_auc, 'Test AUPR: %.4f' % test_aupr,
          'Test Acc: %.4f' % accuracy_test, 'Test Pre: %.4f' % precision_test,
          'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test)
    print('embedding size:', embedding_size, 'epoch:', epoch+1, 'aggregator:', aggregator, 
          'layers:', layers, 'dropout:', dropout, 'lr:', lr, 'weight_decay:', wd)


    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    return train_losses,val_losses
