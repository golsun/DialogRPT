# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb, time, sys, warnings
import numpy as np
from feeder import Feeder
from model import Scorer, JointScorer
import matplotlib.pyplot as plt


class Master:

    def __init__(self, opt):
        self.opt =  opt
        if opt.path_load is not None and (opt.path_load.endswith('.yaml') or opt.path_load.endswith('.yml')):
            self._model = JointScorer(opt)
        else:
            self._model = Scorer(opt)
        if opt.path_load is not None:
            self._model.load(opt.path_load)
        self.parallel()

        if opt.task != 'play':
            if opt.fld_data is not None:
                self.feeder = Feeder(opt)

        if opt.task == 'train':
            opt.save()
            os.makedirs(opt.fld_out + '/ckpt', exist_ok=True)
            self.path_log = self.opt.fld_out + '/log.txt'
        else:
            self.path_log = self.opt.fld_out + '/log_infer.txt'
    
    
    def print(self, s=''):
        try:
            print(s)
        except UnicodeEncodeError:
            print('[UnicodeEncodeError]')
            pass
        with open(self.path_log, 'a', encoding='utf-8') as f:
            f.write(s+'\n')


    def parallel(self):
        if self.opt.cuda:
            self._model = self._model.cuda()
        n_gpu = torch.cuda.device_count()
        if self.opt.cuda and n_gpu > 1:
            print('paralleling on %i GPU'%n_gpu)
            self.model = torch.nn.DataParallel(self._model)
            # after DataParallel, a warning about RNN weights shows up every batch
            warnings.filterwarnings("ignore")
            # after DataParallel, attr of self.model become attr of self.model.module
            self._model = self.model.module
            self.model.core = self.model.module.core
            self.model.tokenizer = self._model.tokenizer
        else:
            self.model = self._model
        if self.opt.task == 'train':
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.opt.lr)
        

    def train(self):
        vali_loss, best_acc = self.vali()
        best_trained = 0
        step = 0
        n_trained = 0
        t0 = time.time()
        
        list_trained = [0]
        list_train_loss = [np.nan]
        list_train_acc = [np.nan]
        list_vali_loss = [vali_loss]
        list_vali_acc = [best_acc]
        acc_history = []

        while step < self.opt.step_max:
            self.model.train()
            self.optimizer.zero_grad()
            batch = self.feeder.get_batch(self.opt.batch)
            pred = self.model.forward(batch)
            loss = self.loss(pred)
            loss = loss.mean()      # in case of parallel-training

            loss.backward()    
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)
            self.optimizer.step()

            acc = (pred > 0.5).float().mean().item()
            acc_history.append(acc)
            if len(acc_history) > self.opt.len_acc:
                acc_history.pop(0)
            avg_train_acc = np.mean(acc_history)
            step += 1
            n_trained += self.opt.batch
            info = 'step %i trained %.3f best %.2f'%(step, n_trained/1e6, best_acc)

            if step % self.opt.step_print == 0:
                speed = (n_trained / 1e6) / ((time.time() - t0) / 3600)

                self.print('%s speed %.2f hr_gap %.2f score_gap %.2f rank_gap %.2f loss %.4f acc %.3f'%(
                    info,
                    speed, 
                    np.median(batch['hr_gap']),
                    (np.array(batch['score_pos']) - np.array(batch['score_neg'])).mean(),
                    (np.array(batch['rank_pos']) - np.array(batch['rank_neg'])).mean(),
                    loss,
                    avg_train_acc,
                    ))

            if step % self.opt.step_vali == 0:
                vali_loss, vali_acc = self.vali(info)
                if vali_acc > best_acc:
                    self.save(self.opt.fld_out + '/ckpt/best.pth')
                    best_acc = vali_acc
                    best_trained = n_trained
                sys.stdout.flush()

                list_trained.append(n_trained/1e6)
                list_train_loss.append(loss.item())
                list_train_acc.append(avg_train_acc)
                list_vali_loss.append(vali_loss)
                list_vali_acc.append(vali_acc)
                _, axs = plt.subplots(3, 1, sharex=True)
                
                axs[0].plot(list_trained, list_train_loss, 'b', label='train')
                axs[0].plot(list_trained, list_vali_loss, 'r', label='vali')
                axs[0].legend(loc='best')
                axs[0].set_ylabel('loss')
                
                axs[1].plot(list_trained, list_train_acc, 'b', label='train')
                axs[1].plot(list_trained, list_vali_acc, 'r', label='vali')
                axs[1].plot([best_trained/1e6, n_trained/1e6], [best_acc, best_acc], 'k:')
                axs[1].set_ylabel('acc')
                
                axs[-1].set_xlabel('trained (M)')
                axs[0].set_title(self.opt.fld_out + '\n' + self.opt.fld_data + '\nbest_acc = %.4f'%best_acc)
                plt.tight_layout()
                plt.savefig(self.opt.fld_out + '/log.png')
                plt.close()

            if step % self.opt.step_save == 0:
                self.save(self.opt.fld_out + '/ckpt/last.pth')


    def loss(self, pred):
        # pred is the probability to pick the positive response, given a context and a negative response
        return - torch.log(pred).mean() 


    def vali(self, info=''):
        n_print = min(self.opt.batch, self.opt.vali_print)
        self.model.eval()
        loss = 0
        acc = 0
        hr_gap = 0
        score_gap = 0
        rank_gap = 0
        n_batch = int(self.opt.vali_size/self.opt.batch)
        self.feeder.reset('vali')
        
        for _ in range(n_batch):
            batch = self.feeder.get_batch(self.opt.batch, sub='vali', 
                    min_score_gap=self.opt.min_score_gap, min_rank_gap=self.opt.min_rank_gap)
            with torch.no_grad():
                pred = self.model.forward(batch)
                loss += self.loss(pred)
            acc += (pred > 0.5).float().mean()
            score_gap += (np.array(batch['score_pos']) - np.array(batch['score_neg'])).mean()
            rank_gap += (np.array(batch['rank_pos']) - np.array(batch['rank_neg'])).mean()
            hr_gap += np.median(batch['hr_gap'])
            
        loss /= n_batch
        acc /= n_batch
        score_gap /= n_batch
        rank_gap /= n_batch
        hr_gap /= n_batch
        s = '%s hr_gap %.2f score_gap %.2f rank_gap %.2f loss %.4f acc %.3f'%(
            info,
            hr_gap,
            score_gap,
            rank_gap,
            loss,
            acc,
            )
        s = '[vali] ' + s.strip()
        if not n_print:
            self.print(s)
            return loss.mean().item(), acc

        with torch.no_grad():
            pred_pos = self.model.core(batch['ids_pos'], batch['len_pos'])
            pred_neg = self.model.core(batch['ids_neg'], batch['len_neg'])

        def to_np(ids):
            if self.opt.cuda:
                ids = ids.cpu()
            return ids.detach().numpy()

        ids_pos = to_np(batch['ids_pos'])
        ids_neg = to_np(batch['ids_neg'])

        for j in range(n_print):
            l_cxt = batch['len_cxt'][j]
            cxt = self.model.tokenizer.decode(ids_pos[j, :l_cxt])
            pos = self.model.tokenizer.decode(ids_pos[j, l_cxt:]).strip('<|ndoftext|>')
            neg = self.model.tokenizer.decode(ids_neg[j, l_cxt:]).strip('<|ndoftext|>')
            self.print(cxt)
            self.print('hr_gap %s'%batch['hr_gap'][j])
            self.print('%s\t%.2f\t%.3f\t%s'%(batch['score_pos'][j], batch['rank_pos'][j], pred_pos[j], pos))
            self.print('%s\t%.2f\t%.3f\t%s'%(batch['score_neg'][j], batch['rank_neg'][j], pred_neg[j], neg))
            self.print()
        
        self.print(s)
        return loss.mean().item(), acc


    def save(self, path):
        torch.save(self._model.state_dict(), path)
        self.print('saved to '+path)
