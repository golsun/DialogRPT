<p align="center">
    <br>
    <img src="doc/icon.png" width="400"/>
    <br>
<p>

# DialogRPT: Dialog Ranking Pretrained Transformers

[DialogRPT](https://arxiv.org/abs/2009.06978/) predicts human feedback (upvotes👍 or replies💬) of dialogue responses. 

It is a set of dialog response ranking models proposed by [Microsoft Research NLP Group](https://www.microsoft.com/en-us/research/group/natural-language-processing/) trained on 100 + millions of human feedback data, accepted to appear at [EMNLP'20](https://2020.emnlp.org/). 
It can be used to improve existing dialog generation model (e.g., [DialoGPT](https://github.com/microsoft/DialoGPT)) by re-ranking the generated response candidates.
This repo provides a PyTorch implementation and pretrained models.

Quick links: 
* [Paper](https://arxiv.org/abs/2009.06978/)
* [Intro Talk](https://slideslive.com/38938970/dialogue-response-ranking-training-with-largescale-human-feedback-data) and [Slides](https://github.com/golsun/DialogRPT/blob/master/doc/DialogRPT-EMNLP.pdf)
* Demo: [original](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing) or [HuggingFace](https://colab.research.google.com/drive/1cAtfkbhqsRsT59y3imjR1APw3MHDMkuV?usp=sharing)
* [Dataset](https://dialogfeedback.github.io/data.html)

We considered the following tasks and provided corresponding pretrained models.
(Click 💾 to download original pytorch checkpoint for this repo, or click 🤗 to use HuggingFace model card)


| Task | Description  | Pretrained model  |
| :------------- | :----------- | :-----------: |
| **Human feedback** | | |
| `updown` |  How likely the response gets the most upvotes? | [💾](https://xiagnlp2.blob.core.windows.net/dialogrpt/updown.pth) / [🤗](https://huggingface.co/microsoft/DialogRPT-updown?text=I+love+NLP%21+<%7Cendoftext%7C>+Me+too%21) |
| `width`| How likely the response gets the most direct replies?  | [💾](https://xiagnlp2.blob.core.windows.net/dialogrpt/width.pth) / [🤗](https://huggingface.co/microsoft/DialogRPT-width?text=I+love+NLP%21+<%7Cendoftext%7C>+Me+too%21) |
| `depth`| How likely the response gets the longest follow-up thread?  | [💾](https://xiagnlp2.blob.core.windows.net/dialogrpt/depth.pth) / [🤗](https://huggingface.co/microsoft/DialogRPT-depth?text=I+love+NLP%21+<%7Cendoftext%7C>+Me+too%21) |
| **Human-like** (human vs fake) | | |
| `human_vs_rand`| How relevant the response is for the given context?  | [💾](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_rand.pth) / [🤗](https://huggingface.co/microsoft/DialogRPT-human-vs-rand?text=I+love+NLP%21+<%7Cendoftext%7C>+Me+too%21) |
| `human_vs_machine`| How likely the response is human-written rather than machine-generated?  | [💾](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_machine.pth) / [🤗](https://huggingface.co/microsoft/DialogRPT-human-vs-machine?text=I+love+NLP%21+<%7Cendoftext%7C>+Me+too%21) |


## Contents:

* [Quick Start](#Quick-Start)
  * [Install](#Install), try [this demo](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing), or use Hugging Face model card with this [demo](https://colab.research.google.com/drive/1cAtfkbhqsRsT59y3imjR1APw3MHDMkuV?usp=sharing)
  * [Use rankers only](#Use-rankers-only): use DialogRPT as evalution metric
  * [Use generator + ranker](#Use-generator-+-ranker): improve generator by reranking hypotheses with DialogRPT
* [Data](#Data)
* [Training](#Training)
* [Evaluation](#Evaluation)
  * [Human feedback prediction](#Human-feedback-prediction)
  * [Human-like classification](#Human-like-classification)
* [Citation](#Citation)




## Quick Start


### Install

**Option 1**: run locally
```
git clone https://github.com/golsun/DialogRPT
cd DialogRPT
conda create -n dialogrpt python=3.6
conda activate dialogrpt
pip install -r requirements.txt
```

**Option 2**: run on Colab Notebook. You can either use [Demo (original)](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing) or [Demo (HuggingFace)](https://colab.research.google.com/drive/1cAtfkbhqsRsT59y3imjR1APw3MHDMkuV?usp=sharing)
<img src="doc/demo.PNG" width="700">


### Use rankers only
In the following example, the model predicts that, given the same context *"I love NLP!"*, response *"Here’s a free textbook (URL) in case anyone needs it."* is gets more upvotes than response *"Me too!"*.
```bash
python src/score.py play -p=restore/updown.pth
#
# Context:  I love NLP!
# Response: Here’s a free textbook (URL) in case anyone needs it.
# score = 0.613

# Context:  I love NLP!
# Response: Me too!
# score = 0.111
```
You can also play the ensemble model, which involves multiple models defined in its [config file](restore/ensemble.yml) (see this file for details). 
```bash
python src/main.py play -p=restore/ensemble.yml
```
To score a list of (context, response) pairs, please provide a input file (`--data`), which is tab-separated in format `context \t response0 \t response1 ...`. See example [input file](https://github.com/golsun/DialogRPT/blob/master/doc/toy.tsv)
* Using a single ranker (see [expected output](https://github.com/golsun/DialogRPT/blob/master/doc/toy.tsv.updown.jsonl))
```bash
python src/score.py test --data=doc/toy.tsv -p=restore/updown.pth
# downloading pretrained model to restore/updown.pth
# 100% [....................] 1520029114 / 1520029114
# loading from restore/updown.pth
# ranking doc/toy.tsv
# totally processed 2 line, avg_hyp_score 0.264, top_hyp_score 0.409
# results saved to doc/toy.tsv.ranked.jsonl
```
* Using an ensemble model (see [expected output](https://github.com/golsun/DialogRPT/blob/master/doc/toy.tsv.ensemble.jsonl))
```bash
python src/score.py test --data=doc/toy.tsv -p=restore/ensemble.yml
```
Statistics of the scoring results can be shown with the following command, e.g. for `doc/toy.tsv.ensemble.jsonl`
```bash
python src/score.py stats --data=doc/toy.tsv.ensemble.jsonl
#                         |best   |avg
# ----------------------------------------
#               _score    |0.339  |0.206
#        human_vs_rand    |0.928  |0.861
#     human_vs_machine    |0.575  |0.525
#               updown    |0.409  |0.264
#                depth    |0.304  |0.153
#                width    |0.225  |0.114
#                final    |0.339  |0.206
# ----------------------------------------
# n_cxt: 2
# avg n_hyp per cxt: 2.50
```


### Use generator + ranker
Dialog generation models can be improved by integrating with the response ranking models.
For example, given the context *"Can we restart 2020?"*, DialoGPT may return the following responses by sampling decoding (or you can try beam search without `--sampling`). Some of them, e.g., *"Yes, we can."* has a high generation probability (`gen 0.496`), but less interesting (`ranker 0.302`). So the rankers will put in position lower than ones more likely to be upvoted, e.g. *"I think we should go back to the beginning, and start from the beginning."* which is relatively less likely to be generated (`gen 0.383`) but seems more interesting (`ranker 0.431`)
```bash
python src/generation.py play -pg=restore/medium_ft.pkl -pr=restore/updown.pth --sampling
#
# Context:        Can we restart 2020?
# 0.431 gen 0.383 ranker 0.431    I think we should go back to the beginning, and start from the beginning.
# 0.429 gen 0.227 ranker 0.429    I think I'll just sit here and wait for 2020
# 0.377 gen 0.249 ranker 0.377    Yeah, let's just start from the beginning
# 0.323 gen 0.195 ranker 0.323    I think we should just give up and let the year just pass.
# 0.304 gen 0.395 ranker 0.304    Yes. We can.
# 0.302 gen 0.496 ranker 0.302    Yes, we can.
# 0.283 gen 0.351 ranker 0.283    It's been a while since we've seen a good reboot.
# 0.174 gen 0.306 ranker 0.174    I'm up for it
# 0.168 gen 0.463 ranker 0.168    I'm down
# 0.153 gen 0.328 ranker 0.153    I think so, yes.
# ...
```
Similarly, you can use the [ensemble model](restore/ensemble.yml).
```
python src/generation.py -pg=restore/medium_ft.pkl -pr=restore/ensemble.yml
```
To generate from a list of contexts stored in a line-separated file, provide it with `--path_test` and use the command below:
```
python src/generation.py test --path_test=path/to/list/of/contexts -pg=restore/medium_ft.pkl -pr=restore/ensemble.yml
```


## Data

As the Pushshift Reddit dataset was deleted from [this server](https://files.pushshift.io/reddit), the data extraction pipeline of this release no longer works. As an alternative, you may want to use the Pushshift [API](https://github.com/pushshift/api).

## Training
We use [DialoGPT](https://github.com/microsoft/DialoGPT) to initialize the model. Please download with
```
wget https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl -P restore
```
For the human feedback prediction tasks, we specify `min_score_gap` and `min_rank_gap` to only validate on less-noisy samples (not applied to training).
```
python src/main.py train --data=data/out/updown -p=restore/medium_ft.pkl --min_score_gap=20 --min_rank_gap=0.5
python src/main.py train --data=data/out/depth -p=restore/medium_ft.pkl --min_score_gap=4 --min_rank_gap=0.5
python src/main.py train --data=data/out/width -p=restore/medium_ft.pkl --min_score_gap=4 --min_rank_gap=0.5
```
For `human_vs_rand` task, use the `--mismatch` flag to feed rand human response as negative examples. We can reuse previous dataset (e.g. `data/out/updown`).
```
python src/main.py train --data=data/out/updown -p=restore/medium_ft.pkl --mismatch
```
For `human_vs_machine` task, we build dataset by pair human response with a response generated by [DialoGPT](https://github.com/microsoft/DialoGPT) with topk decoding
```
python src/main.py train --data=data/out/human_vs_machine -p=restore/medium_ft.pkl
```

We trained all models on a Nvidia V100 4-core GPU (each core with 32G memory) with the following hyperparameters. Checkpoint with the best validation accuracy is used as final model.
| Argument    | Value |  Description |
| :------------- | :-----------: |:------------- | 
| `batch`    | 256 | total batch size for all GPUs. | 
| `vali_size`    | 1024 | number of samples used for validation (i.e. dev set size). | 
| `lr` | 3e-05 | learning rate |
| `max_seq_len` | 50 | max allowed sequence length. <br> if longer, leading tokens will be truncated |
| `max_hr_gap` | 1 | max allowed hour difference between positive and negative samples. <br> If longer, this pair will be discarded for train/vali|


## Evaluation

### Human feedback prediction

The performance on `updown`, `depth`, and `width` can be measured with the following commands, respectively.
The `--min_score_gap` and `--min_rank_gap` arguments are consistent with the values used to measure validation loss during training.
```
python src/score.py eval_human_feedback -p=restore/updown.pth --data=test/human_feedback/updown.tsv --min_score_gap=20 --min_rank_gap=0.5
python src/score.py eval_human_feedback -p=restore/depth.pth --data=test/human_feedback/depth.tsv --min_score_gap=4 --min_rank_gap=0.5
python src/score.py eval_human_feedback -p=restore/width.pth --data=test/human_feedback/width.tsv --min_score_gap=4 --min_rank_gap=0.5
```

The expected pairwise accuracy on 5000 test samples is listed in the table below (from Table 5 of the [paper](https://arxiv.org/abs/2009.06978)). Note even by random guess one can get accuracy of 0.500.
| human feedback     | `updown` | `depth` | `width` |
| :-------------      | :------: |:------------: |:--------: |
| Dialog ppl.         |  0.488   | 0.508         | 0.513     | 
| Reverse dialog ppl. |  0.560   | 0.557         | 0.571     | 
| **DialogRPT** (ours)| **0.683** | **0.695**  | **0.752** | 

### Human-like classification

* `human_vs_rand` task: Although the model is trained on `reddit` corpus only, we measured its **zero-shot** performance on several unseen corpora (`twitter`, `dailydialog` and `personachat`)
```bash
python src/score.py eval_human_vs_rand -p=restore/human_vs_rand.pth --data=test/human_vs_fake/reddit
python src/score.py eval_human_vs_rand -p=restore/human_vs_rand.pth --data=test/human_vs_fake/dailydialog
python src/score.py eval_human_vs_rand -p=restore/human_vs_rand.pth --data=test/human_vs_fake/twitter
python src/score.py eval_human_vs_rand -p=restore/human_vs_rand.pth --data=test/human_vs_fake/personachat
```
The expected `hits@k` metric on 5000 test samples is listed in the table below (from Table 7 of the [paper](https://arxiv.org/abs/2009.06978)).
`hits@k` measures, for the same context, given `k` positive responses and `n` negative responses, how many positive responses are in top-`k` of the ranked responses.
| `human_vs_rand`     | `reddit` | `dailydialog` | `twitter` | `personachat` |
| :-------------      | :------: |:------------: |:--------: |:------------: |
| BM25                |  0.309   | 0.182         | 0.178     | 0.117         |
| Dialog ppl.         |  0.560   | 0.176         | 0.107     | 0.108         |
| Reverse dialog ppl. |  0.775   | 0.457         | 0.440     | 0.449         |
| [ConveRT](https://arxiv.org/abs/1911.03688) |  0.760   | 0.380         | 0.439     | 0.197         |
| **DialogRPT** (ours)| **0.886** | **0.621**  | **0.548** | **0.479**     |

* `human_vs_machine` task: its performance is only evaluated for `reddit` corpus. 
```bash
python src/score.py --task=eval_human_vs_machine -p=restore/human_vs_machine.pth --data=test/human_vs_fake/reddit
# expecting accuracy ~0.98
```


## Citation
If you use our dataset or model, please cite our [paper](https://arxiv.org/abs/2009.06978)

```
@inproceedings{gao2020dialogrpt,
    title={Dialogue Response RankingTraining with Large-Scale Human Feedback Data},
    author={Xiang Gao and Yizhe Zhang and Michel Galley and Chris Brockett and Bill Dolan},
    year={2020},
    booktitle={EMNLP}
}
```
