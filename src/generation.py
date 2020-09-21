# author: Xiang Gao at Microsoft Research AI NLP Group

import torch, pdb
import numpy as np
from shared import download_model

class GPT2Generator:

    def __init__(self, path, cuda):
        from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)        
        self.model = GPT2LMHeadModel(model_config)
        download_model(path)
        weights = torch.load(path)
        if "lm_head.decoder.weight" in weights:
            weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
            weights.pop("lm_head.decoder.weight",None)
        self.model.load_state_dict(weights)
        self.ix_EOS = 50256
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
    

    def predict(self, cxt, topk=3, topp=0.8, beam=10, max_t=30):
        conditioned_tokens = self.tokenizer.encode(cxt) + [self.ix_EOS]
        len_cxt = len(conditioned_tokens)
        tokens = torch.tensor([conditioned_tokens]).view(1, -1)
        if self.cuda:
            tokens = tokens.cuda()
        sum_logP = [0]
        finished = []

        for _ in range(max_t):
            outputs = self.model(tokens)
            predictions = outputs[0]
            logP = torch.log_softmax(predictions[:, -1, :], dim=-1)
            next_logP, next_token = torch.topk(logP, topk)
            sumlogP_ij = []
            sum_prob = 0
            for i in range(tokens.shape[0]):
                for j in range(topk):
                    sum_prob += np.exp(logP[i, j].item())
                    if sum_prob > topp:
                        break
                    if next_token[i, j] == self.ix_EOS:
                        seq = torch.cat([tokens[i, len_cxt:], next_token[i, j].view(1)], dim=-1)
                        if self.cuda:
                            seq = seq.cpu()
                        seq = seq.detach().numpy().tolist()
                        prob = np.exp((sum_logP[i] + next_logP[i, j].item()) / len(seq))
                        hyp = self.tokenizer.decode(seq[:-1])   # don't include EOS
                        finished.append((prob, hyp))
                    else:
                        sumlogP_ij.append((
                            sum_logP[i] + next_logP[i, j].item(), 
                            i, j))
                

            if not sumlogP_ij:
                break
            sumlogP_ij = sorted(sumlogP_ij, reverse=True)[:min(len(sumlogP_ij), beam)]
            new_tokens = []
            new_sum_logP = []
            for _sum_logP, i, j in sumlogP_ij:
                new_tokens.append(
                        torch.cat([tokens[i,:], next_token[i, j].view(1)], dim=-1).view(1, -1)
                        )
                new_sum_logP.append(_sum_logP)
            tokens = torch.cat(new_tokens, dim=0)
            sum_logP = new_sum_logP

        return finished

    def play(self, topk=3, topp=0.8, beam=10):
        while True:
            cxt = input('\ncxt:\t')
            if not cxt:
                break
            ret = self.predict(cxt, topk=topk, topp=topp, beam=beam)
            for prob, hyp in sorted(ret, reverse=True):
                print('%.3f\t%s'%(prob, hyp))


class Integrated:
    def __init__(self, generator, ranker):
        self.generator = generator
        self.ranker = ranker
    
    def predict(self, cxt, topk=3, beam=10, wt_ranker=0.5, max_cxt_turn=2):
        prob_hyp = self.generator.predict(cxt, topk=topk, beam=beam)
        probs = np.array([prob for prob, _ in prob_hyp])
        hyps = [hyp for _, hyp in prob_hyp]
        if wt_ranker > 0:
            scores_ranker = self.ranker.predict(cxt, hyps, max_cxt_turn=max_cxt_turn)
            if isinstance(scores_ranker, dict):
                scores_ranker = scores_ranker['final']
            scores = wt_ranker * scores_ranker + (1 - wt_ranker) * probs
        else:
            scores = probs
        ret = []
        for i in range(len(hyps)):
            ret.append((scores[i], probs[i], scores_ranker[i], hyps[i]))
        ret = sorted(ret, reverse=True)
        return ret


    def play(self, topk=3, topp=0.8, beam=10, wt_ranker=0.5):
        while True:
            cxt = input('\ncxt:\t')
            if not cxt:
                break
            ret = self.predict(cxt, topk=topk, topp=topp, beam=beam, wt_ranker=wt_ranker)
            for final, prob_gen, score_ranker, hyp in ret:
                print('%.3f gen %.3f ranker %.3f\t%s'%(final, prob_gen, score_ranker, hyp))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_generator', '-pg', type=str)
    parser.add_argument('--path_ranker', '-pr', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--beam', type=int, default=3)
    parser.add_argument('--wt_ranker', type=float, default=1.)
    parser.add_argument('--topp', type=float, default=0.8)
    args = parser.parse_args()

    cuda = False if args.cpu else torch.cuda.is_available()
    generator = GPT2Generator(args.path_generator, cuda)
    if args.path_ranker is None:
        generator.play(topk=args.topk, beam=args.beam, topp=args.topp)
    else:
        from score import get_model
        ranker = get_model(args.path_ranker, cuda)
        integrated = Integrated(generator, ranker)
        integrated.play(topk=args.topk, beam=args.beam, topp=args.topp, wt_ranker=args.wt_ranker)