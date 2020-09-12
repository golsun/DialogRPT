import torch, pdb
from shared import _cat_
import numpy as np
from model import OptionInfer, Scorer


def get_model(path, cuda=True):
    model = Scorer(OptionInfer(cuda))
    weights = torch.load(path)
    model.load_state_dict(weights)
    if cuda:
        model.cuda()
    return model


def read_data(path, max_n=-1):
    cxts = dict()
    rsps = dict()
    for i, line in enumerate(open(path, encoding='utf-8')):
        ss = line.strip('\n').split('\t')
        cxt, cxt_id = ss[0].split(_cat_)
        cxt_id = cxt_id.strip()
        cxts[cxt_id] = cxt.strip()
        rsps[cxt_id] = [s.split(_cat_)[0] for s in ss[1:]]
        if i == max_n:
            break
    return cxts, rsps


def predict(model, cxt, hyps, max_cxt_turn=None):
    # split into smaller batch to avoid OOM
    n = len(hyps)
    i0 = 0
    scores = []
    while i0 < n:
        i1 = min(i0 + 32, n)
        _scores = model.predict(cxt, hyps[i0: i1], max_cxt_turn=max_cxt_turn)
        scores.append(_scores)
        i0 = i1
    return np.concatenate(scores)


    
def eval_fake(fld, model, fake, max_n=-1, max_cxt_turn=None):
    """
    for a given context, we rank k real and m fake responses
    if x real responses appeared in topk, define acc as x/k, where k = # of real. 
    for a perfect ranking, x == k and acc == 1. and if k == 1, then this acc is equivalent to the hits@k
    """

    print('evaluating %s'%fld)
    acc = []
    cxts, reals = read_data(fld + '/ref.tsv', max_n=max_n)
    _, fakes = read_data(fld + '/%s.tsv'%fake)

    n = 0
    for cxt_id in reals:
        if cxt_id not in fakes:
            print('[WARNING] could not find fake examples for [%s]'%k)
            continue
        scores = predict(model, cxts[cxt_id], reals[cxt_id] + fakes[cxt_id], max_cxt_turn=max_cxt_turn)
        ix_score = sorted([(scores[i], i) for i in range(len(scores))], reverse=True)
        k = len(reals[cxt_id])
        _acc = np.mean([i < k for _, i in ix_score[:k]])
        acc.append(_acc)
        n += 1
        if n % 10 == 0:
            print('evaluated %i, avg acc %.3f'%(n, np.mean(acc)))

    print('final acc is %.3f based on %i samples'%(n, np.mean(acc)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--fld', type=str, default='eval/reddit')
    parser.add_argument('--max_n', type=int, default=1e6)
    parser.add_argument('--max_cxt_turn', type=int, default=2)
    parser.add_argument('--path_pth', '-p', type=str)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    cuda = False if args.cpu else torch.cuda.is_available()
    model = get_model(args.path_pth, cuda)
    if args.task in ['human_vs_rand', 'human_vs_machine']:
        fake = args.task.split('_')[-1]
        eval_fake(args.fld, model, fake, max_n=args.max_n, max_cxt_turn=args.max_cxt_turn)

