import numpy as np
dir = '../data/'
species = 'Saccharomyces cerevisiae'
SEQ_LEN = 3000
SUB_LEN = 100
n_gene = 6007
n_TF = 212
NUM=int(SEQ_LEN/SUB_LEN)


dna2vec={}
with open(dir + 'DNA2vec/pre-trained DNA-8mers.txt', 'r') as f:
  for line in f:
    k_mer=line[:8]
    line1 = line[9:].split(' ')
    vector = [float(x) for x in line1]
    emb=np.array(vector)
    dna2vec[k_mer] = emb

def chunks(l):
    n = SUB_LEN
    y = [l[i:i+n] for i in range(0, len(l), n)]
    return y

def segmentation(seq):
    if len(seq)<8: return np.array(dna2vec['NNNNNNNN'])
    segments = [seq[j:j+8] for j in range(len(seq)-7)]
    vec = [''.join([str(j) for j in i]) for i in segments]
    for i in range(len(vec)):
        if vec[i] not in dna2vec.keys():
            vec[i]='NNNNNNNN'
    vec = [dna2vec[i] for i in vec]
    y = np.array(vec).mean(axis=0)
    return y

def full_map(x):
    temp = chunks(x)
    temp = map(segmentation,temp)
    temp = tuple(temp)
    temp = np.stack(temp)
    if temp.shape[0]<NUM:
        sup=np.zeros(shape=(NUM-temp.shape[0],SUB_LEN),dtype=np.float32)
        temp=np.vstack((temp,sup))
    return temp

def embeddings(x):
    data = np.array([el for el in map(full_map, x)])

    return data

tg_path = dir+species+'/target_gene.seq'
TF_path = dir+species+'/TF.seq'
gene_Seqs={}
TF_Seqs={}
f=open(tg_path,'r')
for line in f:
    text_line = line.strip('\t').split()
    gene_Seqs[text_line[0]]=text_line[2]

f=open(TF_path,'r')
for line in f:
    text_line = line.strip('\t').split()
    TF_Seqs[text_line[0]]=text_line[2]

tgSeqs = gene_Seqs.values()
tfSeqs = TF_Seqs.values()

tg_data=embeddings(tgSeqs).reshape(n_gene,-1)
TF_data=embeddings(tfSeqs).reshape(n_TF,-1)
np.save(dir+species+'/tg_8mers.npy',tg_data)
np.save(dir+species+'/TF_8mers.npy',TF_data)
