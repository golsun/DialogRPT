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
    if x real responses appeared in topk ranked responses, define acc = x/k, where k = # of real. 
    this can be seen as a generalized version of hits@k
    for a perfect ranking, x == k thus acc == 1. 
    """

    assert(os.path.isdir(fld))
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
        if n == max_n:
            break

    print('final acc is %.3f based on %i samples'%(np.mean(acc), n))



def eval_feedback(path, model, max_n=-1, max_cxt_turn=None, min_rank_gap=0., min_score_gap=0, max_hr_gap=1):
    """
    for a given context, we compare two responses, 
    predict which one got better feedback (greater updown, depth, or width)
    return this pairwise accuracy
    """
    assert(path.endswith('.tsv'))
    assert(min_rank_gap is not None)
    assert(min_score_gap is not None)

    print('evaluating %s'%path)
    acc = []
    n = 0
    for line in open(path, encoding='utf-8'):
        cc = line.strip('\n').split('\t')
        if len(cc) != 11:
            continue
        cxt, pos, neg, _, _, _, hr_gap, pos_score, neg_score, pos_rank, neg_rank = cc
        if float(hr_gap) > max_hr_gap:
            continue
        if float(pos_rank) - float(neg_rank) < min_rank_gap:
            continue
        if int(pos_score) - int(neg_score) < min_score_gap:
            continue

        scores = predict(model, cxt, [pos, neg], max_cxt_turn=max_cxt_turn)
        score_pos = scores[0]
        score_neg = scores[1]
        acc.append(float(score_pos > score_neg))
        n += 1
        if n % 10 == 0:
            print('evaluated %i, avg acc %.3f'%(n, np.mean(acc)))
        if n == max_n:
            break

    print('final acc is %.3f based on %i samples'%(np.mean(acc), n))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--max_cxt_turn', type=int, default=2)
    parser.add_argument('--path_pth', '-p', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max_n', type=int, default=5000)
    parser.add_argument('--min_score_gap', type=int)
    parser.add_argument('--min_rank_gap', type=float)
    args = parser.parse_args()

    cuda = False if args.cpu else torch.cuda.is_available()
    model = get_model(args.path_pth, cuda)
    if args.task in ['human_vs_rand', 'human_vs_machine']:
        fake = args.task.split('_')[-1]
        eval_fake(args.data, model, fake, max_n=max_n, max_cxt_turn=args.max_cxt_turn)

    elif args.task == 'feedback':
        eval_feedback(args.data, model, max_cxt_turn=args.max_cxt_turn, 
            min_rank_gap=args.min_rank_gap, max_n=max_n, min_score_gap=args.min_score_gap)