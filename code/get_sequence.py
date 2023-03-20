import numpy as np
import re

dir = '../data'
species ='Saccharomyces cerevisiae'
complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
output = open(dir + species + "/target_gene.seq", 'w')
offset = 3000

def reverse_complement(seq):
    bases = list(seq)
    bases = reversed([complement.get(base, base) for base in bases])
    bases = ''.join(bases)
    return bases

def get_gene_loc(f):
    tem = []
    seqs={}
    genes={}
    while True:
        text_line = f.readline().strip('\t').split()
        if text_line:
            if text_line[0] == 'DEFINITION':
                tem = []
                c = ' '.join(i for i in text_line[4:])
                seqs[c] = []
                genes[c] = []
            elif text_line[0] == 'gene':
                gene = re.findall('\d+\d*?', text_line[1])
                if re.findall('complement(.*)', text_line[1]):
                    gene.append(1)
                else:
                    gene.append(0)
                if len(gene) == 3:
                    nextline = f.readline().strip('\t').split()
                    name = re.findall('\"(.*)\"', nextline[0])
                    if name != []:
                        gene.append(name[0])
                    nextline = f.readline().strip('\t').split()
                    locus_tag = re.findall('\"(.*)\"', nextline[0])
                    if locus_tag != []:
                        gene.append(locus_tag[0])
                else:
                    gene = []
                if len(gene) > 3:
                    genes[c].append(gene)
            elif text_line[0] == 'ORIGIN':
                if len(genes[c])!=0:
                    while text_line[0] != '//':
                        for i in text_line[1:]:
                            tem.append(i)
                        text_line = f.readline().strip('\t').split()
                    seq = ''.join(i.upper() for i in tem)
                    seqs[c]=seq
            else:
                pass
        elif f.readline().strip('\t').split():
            continue
        else:
            break
    for i,gs in genes.items():
        seq=''.join(seqs[i])
        for g in gs:
            pos = int(g[0])
            pos_b = int(g[1])
            if g[2] == 0:
                s = str(seq[pos - 1:pos - 1 + offset])
            if g[2] == 1:
                if pos_b < offset:
                    s = str(seq[0:pos_b])
                else:
                    s = str(seq[pos_b - offset:pos_b])
                s = reverse_complement(s)
            if len(g) == 4:
                gene_seq = g[-1] + '\t' + g[-1] + '\t' + s + '\n'
                output.write(gene_seq)
            else:
                gene_seq = g[-2] + '\t' + g[-1] + '\t' + s + '\n'
                output.write(gene_seq)
    return genes

f = open(dir + species + '/GCF_000146045.2_R64_genomic.gbff', 'r')
output.close()