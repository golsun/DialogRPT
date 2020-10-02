<p align="center">
    <br>
    <img src="doc/icon.png" width="400"/>
    <br>
<p>

# Dialog Ranking Pretrained Transformers

> How likely a dialog response is upvoted 👍 and/or gets replied 💬? 

This is what **DialogRPT** is learned to predict.
It is a set of dialog response ranking models proposed by [Microsoft Research NLP Group](https://www.microsoft.com/en-us/research/group/natural-language-processing/) trained on 100 + millions of human feedback data. 
It can be used to improve existing dialog generation model (e.g., [DialoGPT](https://github.com/microsoft/DialoGPT)) by re-ranking the generated response candidates.

* [EMNLP'20 Paper](https://arxiv.org/abs/2009.06978/) | [Demo](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing) | [Dataset](https://dialogfeedback.github.io/data.html) | [Slides](https://github.com/golsun/DialogRPT/blob/master/doc/DialogRPT-1page.pdf) | [Related MSR NLP team projects](https://github.com/microsoft/MSR-NLP-Projects)

We considered the following tasks and provided corresponding pretrained models.
|Task | Description  | Pretrained model |
| :------------- | :----------- | :-----------: |
|  **Human feedback**  |  given a context and its two human responses, predict...|
| `updown` |  ... which gets more upvotes? | [download](https://xiagnlp2.blob.core.windows.net/dialogrpt/updown.pth) |
| `width`| ... which gets more direct replies?  | [download](https://xiagnlp2.blob.core.windows.net/dialogrpt/width.pth) |
| `depth`|  ... which gets longer follow-up thread?  | [download](https://xiagnlp2.blob.core.windows.net/dialogrpt/depth.pth) |
|  **Human-like** (human vs fake) | given a context and one human response, distinguish it with... |
| `human_vs_rand`| ... a random human response  | [download](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_rand.pth) |
| `human_vs_machine`| ... a machine generated response  | [download](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_machine.pth) |

## Contents:

* [Quick Start](#Quick-Start)
  * [Install](#Install) or try [this demo](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing)
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

**Option 2**: run on [this Colab Notebook](https://colab.research.google.com/drive/1jQXzTYsgdZIQjJKrX4g3CP0_PGCeVU3C?usp=sharing)
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
For example, given the context *"Can we restart 2020?"*, DialoGPT may return the following responses. Some of them, e.g., "No, we can't." has a high generation probability (`gen 0.314`), but less interesting (`ranker 0.350`). So the rankers will put in position lower than ones more likely to be upvoted, e.g. "No, we can't. It's too late for that. We need to go back in time and start from the beginning of the universe."
```bash
python src/generation.py play -pg=restore/medium_ft.pkl -pr=restore/updown.pth
#
# Context:        Can we restart 2020?
# 0.506 gen 0.210 ranker 0.506    No, we can't. It's too late for that. We need to go back in time and start from the beginning of the universe
# 0.350 gen 0.314 ranker 0.350    No, we can't.
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

Traning dataset can be built with [this script](https://github.com/golsun/DialogRPT/blob/master/data.sh), which downloads raw data from [a third party dump](https://files.pushshift.io/reddit) and extract comparable pairs of comments for classification tasks.
```bash
sh data.sh
```
Testing data can be downloaded [here](https://xiagnlp2.blob.core.windows.net/dialogrpt/test.zip) use the command below
```
wget https://xiagnlp2.blob.core.windows.net/dialogrpt/test.zip
unzip test.zip
```
Please checkout our [Dataset webpage](https://dialogfeedback.github.io/data.html) for data examples, description, statistics and more.

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
