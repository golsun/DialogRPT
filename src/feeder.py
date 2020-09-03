# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb
import numpy as np


class Feeder:
    # load train/vali/test data

    def __init__(self, opt):
        self.opt = opt
        self.files = dict()
        if self.opt.mismatch:
            self.files_mismatch = dict()
        for sub in ['train', 'vali', 'test']:
            self.reset(sub)
        self.ix_EOS = 50256
        self.ix_OMT = 986


    def reset(self, sub):
        print('resetting '+sub)
        path = '%s/%s.tsv'%(self.opt.fld_data, sub)
        if os.path.exists(path):
            self.files[sub] = open(path)
            if self.opt.mismatch:
                self.files_mismatch[sub] = open(path)
                # assuming f is already shuffled, this step makes f and f_mismatch mismatch
                for _ in range(100):
                    self.files[sub].readline()


    def get_batch(self, size, sub='train', min_score_gap=1, min_rank_gap=0):
        ids_pos = []
        len_pos = []
        ids_neg = []
        len_neg = []
        len_cxt = []
        score_pos = []
        score_neg = []
        rank_pos = []
        rank_neg = []
        hr_gap = []
        if sub != 'train':
            np.random.seed(2020)

        def ints(s):
            return [int(x) for x in s.split()]
        def pad(seq):
            return seq + [self.ix_EOS] * (self.opt.max_seq_len - len(seq))

        def read():
            total = 0
            used = 0
            for line in self.files[sub]:
                if line.startswith('#'):
                    continue
                # old data is title + ' . ' + selftext, ' .' is 764 and often used as ' .jpg' thus misleading
                line = line.replace(' 764\t', '\t').replace(' 764 50256 ', ' 50256 ')   
                total += 1
                ss = line.strip('\n').split('\t')
                cxt = ints(ss[0])
                reply_pos = ints(ss[1])
                # _score_pos, _score_neg, _rank_pos, _rank_neg = ss[-4:]
                try:
                    _hr_gap = float(ss[-5])
                except ValueError:
                    _hr_gap = np.nan
                _score_pos = int(ss[-4])
                _rank_pos = float(ss[-2])

                if self.opt.mismatch:
                    _score_neg = np.nan
                    _rank_neg = np.nan
                    line_mismatch = self.files_mismatch[sub].readline()
                    ss_mismatch = line_mismatch.strip('\n').split('\t')
                    reply_neg = ints(ss_mismatch[1])
                    
                else:
                    reply_neg = ints(ss[2])
                    _score_neg = int(ss[-3])
                    _rank_neg = float(ss[-1])
                    if _score_pos - _score_neg < min_score_gap:
                        continue
                    if _rank_pos - _rank_neg < min_rank_gap:
                        continue
                    if self.opt.max_hr_gap > 0 and _hr_gap > self.opt.max_hr_gap:
                        continue
                
                pos = cxt + [self.ix_EOS] + reply_pos
                score_pos.append(_score_pos)
                rank_pos.append(_rank_pos)

                neg = cxt + [self.ix_EOS] + reply_neg                
                score_neg.append(_score_neg)
                rank_neg.append(_rank_neg)
                
                # make sure cxt still same even after cut
                n_del = max(len(pos), len(neg)) - self.opt.max_seq_len
                if n_del > 0:
                    pos = pos[n_del:]
                    neg = neg[n_del:]
                    cxt = cxt[n_del:]

                len_cxt.append(len(cxt))
                len_pos.append(len(pos))
                len_neg.append(len(neg))
                ids_pos.append(pad(pos))
                ids_neg.append(pad(neg))
                hr_gap.append(_hr_gap)

                used += 1
                if len(ids_pos) == size:
                    break
                
        while True:
            read()
            if len(ids_pos) == size:
                break
            self.reset(sub)
        
        ids_pos = torch.LongTensor(ids_pos)
        ids_neg = torch.LongTensor(ids_neg)
        if self.opt.cuda:
            ids_pos = ids_pos.cuda()
            ids_neg = ids_neg.cuda()
        return {
            'ids_pos':ids_pos,
            'ids_neg':ids_neg,
            'len_pos':len_pos,
            'len_neg':len_neg,
            'len_cxt':len_cxt,
            'score_pos': score_pos,
            'score_neg': score_neg,
            'rank_pos': rank_pos,
            'rank_neg': rank_neg,
            'hr_gap': hr_gap,
            }