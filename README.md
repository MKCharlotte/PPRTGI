# PPRTGI: A novel graph model based on personalized PageRank for TF-target gene interaction prediction with sequence and genetic information
Transcription factors (TFs) regulation is required for the vast majority of biological processes in living organisms. Some diseases may be caused by improper transcriptional regulation. Identifying the target genes of TFs is thus critical for understanding cellular processes and analyzing disease molecular mechanisms. Computational approaches can be challenging for one to employ when attempting to predict potential interactions between TFs and target genes. In this paper, we present a novel graph model (PPRTGI) for detecting TF-target gene interactions using DNA sequence features. Feature representations of TFs and target genes are extracted from sequence embeddings and biological associations. Then, by combining the aggregated node feature with graph structure, PPRTGI uses a graph neural network with personalized PageRank to learn interaction patterns. Finally, a bilinear decoder is applied to predict interaction scores between TF and target gene nodes. We designed experiments on six datasets from different species. The experimental results show that PPRTGI is effective in regulatory interaction inference, with our proposed model achieving an area under receiver operating characteristic score of 93.87% and an area under precision-recall curves score of 88.79% on the human dataset. This paper proposes a new method for predicting TF-target gene interactions, which provides new insights into modeling molecular networks and can thus be used to gain a better understanding of complex biological systems.

![model](https://user-images.githubusercontent.com/53011248/226361040-26948259-8269-4c2d-9e69-b6769e895103.png)

## Requirements
* python=3.7.0
* dgl-cu102==0.8.0.post1
* mxnet-cu102==1.6.0.post0
* node2vec==0.4.3
* numpy=1.19.2


