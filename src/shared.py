# author: Xiang Gao at Microsoft Research AI NLP Group


import re, pdb
import scipy.stats
_cat_ = ' <-COL-> '


def spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]

def get_scale():
    scale = dict()
    for line in open('scale.tsv'):
        line = line.strip('\n')
        if line.startswith('#') or len(line) == 0:
            continue
        k, s = line.split('\t')
        scale[k] = float(s)
    return scale