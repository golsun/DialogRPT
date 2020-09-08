# author: Xiang Gao at Microsoft Research AI NLP Group


import re, pdb
import scipy.stats
_cat_ = ' <-COL-> '


def spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]
