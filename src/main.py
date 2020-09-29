# author: Xiang Gao at Microsoft Research AI NLP Group


import argparse, torch, time, pdb
import os, socket
from master import Master


class Option:

    def __init__(self, args):        
        if args.cpu or not torch.cuda.is_available():
            self.cuda = False 
        else:
            self.cuda = True
        self.task = args.task
        self.path_load = args.path_load
        self.batch = args.batch
        self.vali_size = max(self.batch, args.vali_size)
        self.vali_print = args.vali_print
        self.lr = args.lr
        self.max_seq_len = args.max_seq_len
        self.min_score_gap = args.min_score_gap
        self.min_rank_gap = args.min_rank_gap
        self.max_hr_gap = args.max_hr_gap
        self.mismatch = args.mismatch
        self.fld_data = args.data
        if args.task == 'train' or self.path_load is None:
            self.fld_out = 'out/%i'%time.time()
        else:
            self.fld_out = 'out/temp'
        os.makedirs(self.fld_out, exist_ok=True)

        self.clip = 1
        self.step_max = 1e6
        self.step_print = 10
        self.step_vali = 100
        self.step_save = 500
        self.len_acc = self.step_vali


    def save(self):
        d = self.__dict__
        lines = []
        for k in d:
            lines.append('%s\t%s'%(k, d[k]))
        with open(self.fld_out + '/opt.tsv', 'w') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--vali_size', type=int, default=1024)
    parser.add_argument('--vali_print', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--path_load','-p', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--mismatch', action='store_true')
    parser.add_argument('--min_score_gap', type=int)
    parser.add_argument('--min_rank_gap', type=float)
    parser.add_argument('--max_hr_gap', type=float, default=1)
    args = parser.parse_args()

    opt = Option(args)
    master = Master(opt)
    if args.task == 'train':
        master.train()
    elif args.task == 'vali':
        master.vali()