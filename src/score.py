import torch, pdb, os, json
from shared import _cat_
import numpy as np
from model import OptionInfer, Scorer


def get_model(path, cuda=True):
    model = Scorer(OptionInfer(cuda))
    model.load(path)
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
            ss0 =  ss[0].split(_cat_)
            if len(ss0) == 2:
                cxt, cxt_id = ss0
                cxt_id = cxt_id.strip()
            else:
                cxt = ss0[0]
                cxt_id = cxt.strip().replace(' ','')
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
            print('[WARNING] could not find fake examples for [%s]'%cxt_id)
            #pdb.set_trace()
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



def rank_hyps(path, model, max_n=-1, max_cxt_turn=None):
    """
    rank the responses for each given cxt with model
    path is the input file, where in each line, 0-th column is the context, and the rest are responses
    output a jsonl file, and can be read with function `read_ranked_jsonl`
    """

    print('ranking %s'%path)
    lines = []
    n = 0
    sum_avg_score = 0
    sum_top_score = 0
    for i, line in enumerate(open(path, encoding='utf-8')):
        cc = line.strip('\n').split('\t')
        if len(cc) < 2:
            print('[WARNING] line %i only has %i columns, ignored'%(i, len(cc)))
            continue
        cxt = cc[0]
        hyps = cc[1:]
        scores = predict(model, cxt, hyps, max_cxt_turn=max_cxt_turn)
        d = {'line_id':i, 'cxt': cxt}
        scored = []
        for j, hyp in enumerate(hyps):
            scored.append((float(scores[j]), hyp))
        d['hyps'] = list(sorted(scored, reverse=True))
        lines.append(json.dumps(d))
        sum_avg_score += np.mean(scores)
        sum_top_score += np.max(scores)
        n += 1

        if n % 10 == 0:
            print('processed %i line, avg_hyp_score %.3f, top_hyp_score %.3f'%(
                    n, 
                    sum_avg_score/n,
                    sum_top_score/n,
                    ))
        if n == max_n:
            break
    print('totally processed %i line, avg_hyp_score %.3f, top_hyp_score %.3f'%(
                    n, 
                    sum_avg_score/n,
                    sum_top_score/n,
                    ))
    path_out = path+'.ranked.jsonl'
    with open(path_out, 'w') as f:
        f.write('\n'.join(lines))
    print('results saved to '+path_out)
    print('results can be read with function `read_ranked_jsonl`')
    data = read_ranked_jsonl(path_out)
    pdb.set_trace()


def read_ranked_jsonl(path):
    """ read the jsonl file ouput by function rank_hyps"""
    return [json.loads(line) for line in open(path, encoding="utf-8")]



def play(model, max_cxt_turn=None):
    from shared import EOS_token
    model.eval()
    print('enter empty to stop')
    print('use `%s` to delimite turns for a multi-turn context'%EOS_token)
    while True:
        print()
        cxt = input('Context:  ')
        if not cxt:
            break
        hyp = input('Response: ')
        if not hyp:
            break
        score = model.predict(cxt, [hyp], max_cxt_turn=max_cxt_turn)
        if isinstance(score, dict):
            ss = ['%s = %.3f'%(k, score[k][0]) for k in score]
            print(', '.join(ss))
        else:
            print('score = %.3f'%score[0])


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
    if args.task in ['eval_human_vs_rand', 'eval_human_vs_machine']:
        fake = args.task.split('_')[-1]
        eval_fake(args.data, model, fake, max_n=args.max_n, max_cxt_turn=args.max_cxt_turn)

    elif args.task == 'eval_human_feedback':
        eval_feedback(args.data, model, max_cxt_turn=args.max_cxt_turn, 
            min_rank_gap=args.min_rank_gap, max_n=args.max_n, min_score_gap=args.min_score_gap)

    elif args.task == 'rank':
        rank_hyps(args.data, model, max_n=args.max_n, max_cxt_turn=args.max_cxt_turn)

    elif args.task == 'play':
        play(model, max_cxt_turn=args.max_cxt_turn)
