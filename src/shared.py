# author: Xiang Gao at Microsoft Research AI NLP Group

_cat_ = ' <-COL-> '
EOS_token = '_EOS_'


def download_model(path):
    if path is None:
        return
    import os, subprocess
    if os.path.exists(path):
        return
    links = dict()
    for k in ['updown', 'depth', 'width', 'human_vs_rand', 'human_vs_machine']:
        links['restore/%s.pth'%k] = 'https://xiagnlp2.blob.core.windows.net/dialogrpt/%s.pth'%k
    if path not in links:
        return
    cmd = [ 'wget', links[path], '-P', 'restore']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.communicate()