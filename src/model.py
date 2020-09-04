# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from shared import get_scale


class OptionInfer:
    def __init__(self):
        self.cuda = True


class ScorerBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ix_EOS = 50256
        self.ix_OMT = 986


    def core(self, ids, l_ids, return_logits=False, scale=1):
        # to be implemented in child class
        return 0

    
    def predict(self, cxt, hyps, max_cxt_turn=None, scale=1):
        # cxt = str
        # hyps = list of str

        self.eval()
        cxt_turns = cxt.split('_EOS_')
        if max_cxt_turn is not None:
            cxt_turns = cxt_turns[-min(max_cxt_turn, len(cxt_turns)):]
        ids_cxt = []
        for turn in cxt_turns:
            ids_cxt += self.tokenizer.encode(turn.strip()) + [self.ix_EOS]
        seqs = []
        lens = []
        for hyp in hyps:
            seq = ids_cxt + self.tokenizer.encode(hyp.strip())
            lens.append(len(seq))
            seqs.append(seq)
        max_len = max(lens)
        ids = []
        for seq in seqs:
            ids.append(seq + [self.ix_EOS] * (max_len - len(seq)))
        ids = torch.LongTensor(ids)
        if self.opt.cuda:
            ids = ids.cuda()
        scores = self.core(ids, lens, scale=scale)
        if self.opt.cuda:
            scores = scores.cpu()
        return scores.detach().numpy()


    def forward(self, batch):
        logits_pos = self.core(batch['ids_pos'], batch['len_pos'], return_logits=True)
        logits_neg = self.core(batch['ids_neg'], batch['len_neg'], return_logits=True)
        # softmax to get the `probability` to rank pos/neg correctly
        return torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))



class Scorer(ScorerBase):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        n_embd = 1024
        config = GPT2Config(n_embd=n_embd, n_layer=24, n_head=16)
        self.transformer = GPT2Model(config)
        self.score = torch.nn.Linear(n_embd, 1, bias=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        

    def core(self, ids, l_ids, return_logits=False, scale=1):
        n = ids.shape[0]
        attention_mask = torch.ones_like(ids)
        for i in range(n):
            attention_mask[i, l_ids[i]:] *= 0
        hidden_states, _ = self.transformer(ids, attention_mask=attention_mask)
        logits = self.score(hidden_states).squeeze(-1)
        logits = torch.stack([logits[i, l_ids[i] - 1] for i in range(n)])
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits * scale)

    
    def load(self, path):
        print('loading from '+path)
        weights = torch.load(path)
        self.load_state_dict(weights)


class JointScorer(ScorerBase):
    def __init__(self):
        super().__init__()
        import socket
        self.hostname = socket.gethostname()
        print(get_scale())

    
    def _core(self, k, ids, l_ids, scale=1):
        i = self.kk.index(k)
        attr = 'scorer%i'%i
        scorer = getattr(self, attr)
        return scorer.core(ids, l_ids, scale=scale)


    def core(self, ids, l_ids, return_logits=False):
        d_scale = get_scale()
        if self.kk_cond:
            prior = 0
            for k in self.kk_prior:
                scale = d_scale.get(k, 1)
                prior = prior + self._core(k, ids, l_ids, scale=scale)
            prior = prior / len(self.kk_prior)
        else:
            prior = 1

        if self.kk_cond:
            cond = 0
            for k in self.kk_cond:
                scale = d_scale.get(k, 1)
                cond = cond + self._core(k, ids, l_ids, scale=scale)
            cond = cond / len(self.kk_cond)
        else:
            cond = 1
        
        return prior * cond


    def create(self, k):
        path = 'restore/%s/model.pth'%k
        if 'MININT-3LHNLKS' in self.hostname:
            path = path.replace('restore/', 'F:/restore/DialogScorer/')
        print('loading from', path)
        scorer = Scorer(OptionInfer())
        weights = torch.load(path)
        scorer.load_state_dict(weights)
        return scorer

    
    def load(self, paths):
        kk_prior, kk_cond = paths.split('##')
        self.kk_prior = kk_prior.split('#')
        self.kk_cond = kk_cond.split('#')
        prior = ' + '.join(self.kk_prior)
        cond = ' + '.join(self.kk_cond)
        print('==== FORMULA ====')
        print(prior + ' * ' + cond)
        print('=================')

        self.scorers = dict()
        self.kk = list(set(self.kk_prior + self.kk_cond))
        for i, k in enumerate(self.kk):
            attr = 'scorer%i'%i
            scorer = self.create(k)
            setattr(self, attr, scorer)






