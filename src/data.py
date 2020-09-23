# author: Xiang Gao at Microsoft Research AI NLP Group


import bz2, json, os, pickle, pdb, time, random
import urllib.request
import numpy as np
from shared import _cat_


def valid_sub(sub):
    if sub.upper() in [
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 
            'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']:
        # not allowed by Windows system
        return False
    if ':' in sub:
        return False
    return True


def get_dates(year_from, year_to=None):
    if year_to is None:
        year_to = year_from
    dates = []
    for year in range(year_from, year_to + 1):
        for _mo in range(1, 12 + 1):
            mo = str(_mo)
            if len(mo) == 1:
                mo = '0' + mo
            dates.append(str(year) + '-' + mo)
    return dates


def extract_rc(date):
    path_bz2 = '%s/RC_%s.bz2'%(fld_bz2, date)
    nodes = dict()
    edges = dict()
    subs = set()
    n = 0
    m = 0
    kk = ['body', 'link_id', 'name', 'parent_id', 'subreddit']

    def save(nodes, edges):
        for sub in nodes:
            fld = fld_jsonl + '/' + sub
            try:
                os.makedirs(fld, exist_ok=True)
            except NotADirectoryError as e:
                print(e)
                continue
            if sub not in subs:
                open(fld + '/%s_nodes.jsonl'%date, 'w', encoding="utf-8")
                open(fld + '/%s_edges.tsv'%date, 'w', encoding="utf-8")
                subs.add(sub)
            with open(fld + '/%s_nodes.jsonl'%date, 'a', encoding="utf-8") as f:
                f.write('\n'.join(nodes[sub]) + '\n')
            with open(fld + '/%s_edges.tsv'%date, 'a', encoding="utf-8") as f:
                f.write('\n'.join(edges[sub]) + '\n')

    for line in bz2.open(path_bz2, 'rt', encoding="utf-8"):
        n += 1
        line = line.strip('\n')
        try:
            node = json.loads(line)
        except Exception:
            continue
        
        ok = True
        for k in kk:
            if k not in node:
                ok = False
                break
        if not ok:
            break

        if not valid_sub(node['subreddit']):
            continue

        if node['subreddit'] not in nodes:
            nodes[node['subreddit']] = []
            edges[node['subreddit']] = []
        nodes[node['subreddit']].append(line)
        edges[node['subreddit']].append('%s\t%s\t%s'%(node['link_id'], node['parent_id'], node['name']))

        m += 1
        if m % 1e5 == 0:
            save(nodes, edges)
            print('[RC_%s] saved %.2f/%.2f M, %i subreddits'%(date, m/1e6, n/1e6, len(subs)))
            nodes = dict()
            edges = dict()
    
    save(nodes, edges)
    print('[RC_%s] FINAL %.2f/%.2f M, %i subreddits ================'%(date, m/1e6, n/1e6, len(subs)))
    with open(fld_jsonl + '/readme.txt', 'a', encoding='utf-8') as f:
        f.write('[%s] saved %i/%i\n'%(date, m, n))


def extract_rs(date):
    path_bz2 = '%s/RS_%s.bz2'%(fld_bz2, date)
    roots = dict()
    subs = set()
    n = 0
    m = 0
    kk = ['selftext', 'id', 'title', 'subreddit']

    def save(roots):
        for sub in roots:
            fld = fld_jsonl + '/' + sub
            try:
                os.makedirs(fld, exist_ok=True)
            except NotADirectoryError as e:
                print(e)
                continue
            if sub not in subs:
                open(fld + '/%s_roots.jsonl'%date, 'w', encoding="utf-8")
                subs.add(sub)
            with open(fld + '/%s_roots.jsonl'%date, 'a', encoding="utf-8") as f:
                f.write('\n'.join(roots[sub]) + '\n')

    for line in bz2.open(path_bz2, 'rt', encoding="utf-8"):
        n += 1
        line = line.strip('\n')
        try:
            root = json.loads(line)
        except Exception:
            continue
        
        ok = True
        for k in kk:
            if k not in root:
                ok = False
                break
        if not ok:
            break
        if not valid_sub(root['subreddit']):
            continue

        # some bz2, e.g. 2012-09, doesn't have the `name` entry
        if 'name' not in root:
            root['name'] = 't3_' + root['id']

        if root['subreddit'] not in roots:
            roots[root['subreddit']] = []
        roots[root['subreddit']].append(line)

        m += 1
        if m % 1e4 == 0:
            save(roots)
            print('[RS_%s] saved %.2f/%.2f M, %i subreddits'%(date, m/1e6, n/1e6, len(subs)))
            roots = dict()
    
    save(roots)
    print('[RS_%s] FINAL %.2f/%.2f M, %i subreddits ================'%(
        date, m/1e6, n/1e6, len(subs)))
    with open(fld_jsonl + '/readme_roots.txt', 'a', encoding='utf-8') as f:
        f.write('[%s] saved %i/%i\n'%(date, m, n))



def extract_txt(sub, year, tokenizer, overwrite=False, max_subword=3):
    fld = '%s/%s'%(fld_subs, sub)
    os.makedirs(fld, exist_ok=True)
    path_out = '%s/%i_txt.tsv'%(fld, year)
    path_done = path_out + '.done'
    if not overwrite and os.path.exists(path_done):
        return

    dates = get_dates(year)
    open(path_out, 'w', encoding='utf-8')

    def clean(txt):
        if txt.strip() in ['[deleted]', '[removed]']:
            return None
        if '>' in txt or '&gt;' in txt:		# no comment in line ('&gt;' means '>')
            return None

        # deal with URL
        txt = txt.replace('](','] (')
        ww = []
        for w in txt.split():
            if len(w) == 0:
                continue
            if '://' in w.lower() or 'http' in w.lower():
                ww.append('(URL)')
            else:
                ww.append(w)
        if not ww:
            return None
        if len(ww) > 30:        # focus on dialog, so ignore long txt
            return None
        if len(ww) < 1:
            return None
        txt = ' '.join(ww)
        for c in ['\t', '\n', '\r']:     # delimiter or newline
            txt = txt.replace(c, ' ')

        ids = tokenizer.encode(txt)
        if len(ids) / len(ww) > max_subword:      # usually < 1.5. too large means too many unknown words
            return None
        
        ids = ' '.join([str(x) for x in ids])
        return txt, ids

    lines = []
    m = 0
    n = 0
    name_set = set()
    for date in dates:
        path = '%s/%s/%s_nodes.jsonl'%(fld_jsonl, sub, date)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding='utf-8'):
            n += 1
            d = json.loads(line.strip('\n'))
            if d['name'] in name_set:
                continue
            name_set.add(d['name'])
            txt_ids = clean(d['body'])
            if txt_ids is not None:
                txt, ids = txt_ids
                lines.append('%s\t%s\t%s'%(d['name'], txt, ids))
                m += 1
                if m % 1e4 == 0:
                    with open(path_out, 'a', encoding='utf-8') as f:
                        f.write('\n'.join(lines) + '\n')
                    lines = []

    for date in dates:
        path = '%s/%s/%s_roots.jsonl'%(fld_jsonl, sub, date)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding='utf-8'):
            n += 1
            d = json.loads(line.strip('\n'))
            if 'name' not in d:
                d['name'] = 't3_' + d['id']
            if d['name'] in name_set:
                continue
            name_set.add(d['name'])
            txt_ids = clean(d['title'] + ' ' + d['selftext'])
            if txt_ids is not None:
                txt, ids = txt_ids
                lines.append('%s\t%s\t%s'%(d['name'], txt, ids))
                m += 1
                if m % 1e4 == 0:
                    with open(path_out, 'a', encoding='utf-8') as f:
                        f.write('\n'.join(lines) + '\n')
                    lines = []
    if lines:
        with open(path_out, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    s = '[%s %s] txt kept %i/%i'%(sub, year, m, n)
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)


def extract_trees(sub, year):
    fld = '%s/%s'%(fld_subs, sub)
    os.makedirs(fld, exist_ok=True)
    path_out = '%s/%i_trees.pkl'%(fld, year)
    if os.path.exists(path_out):
        return

    trees = dict()
    n = 0
    for date in get_dates(year):
        path = '%s/%s/%s_edges.tsv'%(fld_jsonl, sub, date)
        if not os.path.exists(path):
            #print('no such file: '+path)
            continue
        for line in open(path, encoding='utf-8'):
            n += 1
            link, parent, child = line.strip('\n').split('\t')
            if link not in trees:
                trees[link] = dict()
            trees[link][(parent, child)] = date
    
    if not trees:
        return

    print('[%s %i] %i trees %.1f nodes/tree'%(sub, year, len(trees), n/len(trees)))
    os.makedirs(fld, exist_ok=True)
    pickle.dump(trees, open(path_out, 'wb'))


def extract_time(sub, year, overwrite=False):
    fld = '%s/%s'%(fld_subs, sub)
    os.makedirs(fld, exist_ok=True)
    path_out = '%s/%i_time.tsv'%(fld, year)
    path_done = path_out + '.done'
    if not overwrite and os.path.exists(path_done):
        return
    dates = get_dates(year)
    suffix = 'nodes'
    os.makedirs(fld, exist_ok=True)
    open(path_out, 'w', encoding='utf-8')

    lines = []
    m = 0
    n = 0
    name_set = set()
    for date in dates:
        path = '%s/%s/%s_%s.jsonl'%(fld_jsonl, sub, date, suffix)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding='utf-8'):
            n += 1
            d = json.loads(line.strip('\n'))
            if 'name' not in d:
                d['name'] = 't3_' + d['id']
            if d['name'] in name_set:
                continue
            name_set.add(d['name'])
            t = d['created_utc']
            lines.append('%s\t%s'%(d['name'], t))
            m += 1
            if m % 1e4 == 0:
                with open(path_out, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
                lines = []
    with open(path_out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    s = '[%s %s] time kept %i/%i'%(sub, year, m, n)
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)



def calc_feedback(sub, year, overwrite=False):
    fld = '%s/%s'%(fld_subs, sub)
    path_out = '%s/%i_feedback.tsv'%(fld, year)
    path_done = path_out + '.done'
    if not overwrite and os.path.exists(path_done):
        return

    path_pkl = '%s/%i_trees.pkl'%(fld, year)
    if not os.path.exists(path_pkl):
        return
    trees = pickle.load(open(path_pkl, 'rb'))
    if not trees:
        return

    dates = get_dates(year)
    updown = dict()
    for date in dates:
        path = '%s/%s/%s_nodes.jsonl'%(fld_jsonl, sub, date)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding='utf-8'):
            d = json.loads(line.strip('\n'))
            updown[d['name']] = d['ups'] - d['downs']
        
    if not updown:
        print('empty updown:')
        return

    with open(path_out, 'w', encoding='utf-8') as f:
        f.write('\t'.join(['#path', 'vol', 'width', 'depth', 'updown']) + '\n')

    print('[%s %s] calculating scores for %i trees'%(sub, year, len(trees)))

    n_tree = 0
    n_node = 0
    for root in trees:
        tree = trees[root]
        children = dict()
        for parent, child in tree:
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
        if root not in children:
            continue

        # BFS to get all paths from root to leaf
        q = [[root]]
        paths = []
        while q:
            qsize = len(q)
            for _ in range(qsize):
                path = q.pop(0)
                head = path[-1]
                if head not in children:    # then head is a leaf
                    paths.append(path)
                    continue
                for child in children[head]:
                    q.append(path + [child])
    
        prev = dict()
        for path in paths:
            for i in range(1, len(path)):
                prev[path[i]] = ' '.join(path[:i + 1])

        descendant = dict()
        longest_subpath = dict()
        while paths:
            path = paths.pop(0)
            node = path[0]
            if node not in descendant:
                descendant[node] = set()
                longest_subpath[node] = 0
            descendant[node] |= set(path[1:])
            longest_subpath[node] = max(longest_subpath[node], len(path) - 1)
            if len(path) > 1:
                paths.append(path[1:])

        sorted_nodes = sorted([(len(prev[node].split()), prev[node], node) for node in prev])
        if not sorted_nodes:
            continue

        n_tree += 1
        lines = []
        for _, _, node in sorted_nodes:
            if node == root:
                continue
            if node not in updown:
                continue
            n_node += 1
            lines.append('%s\t%i\t%i\t%i\t%i'%(
                prev[node],                     # turns:    path from its root to this node
                len(descendant[node]),          # vol:      num of descendants of this node
                len(children.get(node, [])),    # width:    num of direct childrent of this node
                longest_subpath[node],          # depth:    num of longest subpath of this node
                updown[node],                   # updown:   `upvotes - downvotes` of this node
                ))
        with open(path_out, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        
    if n_tree:
        s = '[%s %s] %i tree %i nodes'%(sub, year, n_tree, n_node)
    else:
        s = '[%s %s] trees are empty'%(sub, year)
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)


def create_pairs(year, sub, feedback, overwrite=False):
    fld = '%s/%s'%(fld_subs, sub)
    path_out = '%s/%i_%s.tsv'%(fld, year, feedback)
    path_done = path_out + '.done'
    if not overwrite and os.path.exists(path_done):
        return

    ix_feedback = ['vol', 'width', 'depth', 'updown'].index(feedback) + 1
    path_in = '%s/%i_feedback.tsv'%(fld, year)
    if not os.path.exists(path_in):
        return

    time = dict()
    path_time = '%s/%i_time.tsv'%(fld, year)
    if not os.path.exists(path_time):
        return
    for line in open(path_time):
        ss = line.strip('\n').split('\t')
        if len(ss) == 2:
            name, t = ss
            time[name] = int(t)

    open(path_out, 'w', encoding='utf-8')
    print('[%s %s] creating pairs...'%(sub, year))

    def match_time(replies, cxt):
        scores = sorted(set([score for score, _ in replies]))
        m = len(scores)
        if m < 2:
            return 0    # can't create pairs if m < 2
        cand = []
        for score, reply in replies:
            if reply not in time:
                continue
            cand.append((time[reply], score, reply))
        cand = sorted(cand)
        rank = [scores.index(score) / (m - 1) for _, score, _ in cand]
        lines = []
        for i in range(len(cand) - 1):
            t_a, score_a, a = cand[i]
            t_b, score_b, b = cand[i + 1]
            rank_a = rank[i]
            rank_b = rank[i + 1]
            if score_a == score_b:
                continue
            hr = (t_b - t_a)/3600
            if score_b > score_a:
                score_a, score_b = score_b, score_a
                a, b = b, a
                rank_a, rank_b = rank_b, rank_a
            lines.append('\t'.join([
                cxt,
                a,
                b,
                '%.2f'%hr,
                '%i'%score_a,
                '%i'%score_b,
                '%.4f'%rank_a,
                '%.4f'%rank_b,
                ]))
        #pdb.set_trace()
        if lines:
            with open(path_out, 'a') as f:
                f.write('\n'.join(lines) + '\n')
        return len(lines)

    n_line = 0
    prev = None
    replies = []
    for line in open(path_in):
        if line.startswith('#'):
            continue
        ss = line.strip('\n').split('\t')
        turns = ss[0].split()       # including both cxt and resp
        if len(turns) < 2:
            continue
        reply = turns[-1]
        try:
            score = int(ss[ix_feedback])
        except ValueError:
            continue
        parent = turns[-2]
        if parent == prev:
            replies.append((score, reply))
        else:
            if replies:
                n_line += match_time(replies, cxt)
            cxt = ' '.join(turns[:-1])
            prev = parent
            replies = [(score, reply)]
    if replies:
        n_line += match_time(replies, cxt)
        
    s = '[%s %s %s] %i pairs'%(sub, year, feedback, n_line)
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)


def add_seq(sub, year, feedback, overwrite=False):
    fname = '%i_%s'%(year, feedback)
    fld = '%s/%s'%(fld_subs, sub)
    turn_sep = ' 50256 '
    path_out = fld + '/%s_ids.tsv'%fname
    path_done = path_out + '.done'

    if os.path.exists(path_done) and not overwrite:
        return
    if not os.path.exists(fld + '/%s.tsv'%fname):
        return
    
    seq = dict()
    path = '%s/%i_txt.tsv'%(fld, year)
    if not os.path.exists(path):
        return
    for line in open(path, encoding='utf-8'):
        ss = line.strip('\n').split('\t')
        if len(ss) != 3:
            continue
        name, txt, ids = ss
        seq[name] = ids
    print('loaded %i seq'%len(seq))
    open(path_out, 'w', encoding='utf-8')
    print('[%s %s %s] adding seq'%(sub, year, feedback))
    path = fld + '/%s.tsv'%fname
    lines = []
    n = 0
    m = 0
    for line in open(path, encoding='utf-8'):
        line = line.strip('\n')
        if line.startswith('#'):
            continue
    
        n += 1
        ss = line.split('\t')
        if len(ss) < 7:
            continue
        name_cxt, name_pos, name_neg = ss[:3]

        cxt = []
        ok = True
        for name in name_cxt.split():
            if name in seq:
                cxt.append(seq[name])
            else:
                ok = False
                break
        if not ok:
            continue
        cxt = turn_sep.join(cxt)

        if name_pos in seq:
            reply_pos = seq[name_pos]
        else:
            continue
        if name_neg in seq:
            reply_neg = seq[name_neg]
        else:
            continue
        
        lines.append('\t'.join([
            cxt, reply_pos, reply_neg, 
            name_cxt, name_pos, name_neg, 
            ] + ss[3:]))
        m += 1
        if m % 1e4 == 0:
            with open(path_out, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            lines = []

    with open(path_out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    s = '[%s %s %s] pair seq %i/%i'%(sub, year, feedback, m, n)
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)


def combine_sub(year_from, year_to, feedback, overwrite=False, skip_same_pos=True):
    fld = '%s/%s'%(fld_out, feedback)
    os.makedirs(fld, exist_ok=True)
    path_out = fld + '/raw.tsv'
    path_done = path_out + '.done'
    if os.path.exists(path_done) and not overwrite:
        return path_out

    subs = sorted(os.listdir(fld_subs))
    open(path_out, 'w', encoding='utf-8')
    lines = []
    n = 0
    empty = True
    non_empty_subreddits = 0
    for sub in subs:
        empty = True
        for year in range(year_from, year_to + 1):
            path = '%s/%s/%i_%s_ids.tsv'%(fld_subs, sub, year, feedback)
            if not os.path.exists(path):
                continue
            for line in open(path, encoding='utf-8'):
                if line.startswith('#'):
                    continue
                line = line.strip('\n')
                if not line:
                    continue
                lines.append(line)
                empty = False
                n += 1
                if n % 1e5 == 0:
                    with open(path_out, 'a', encoding='utf-8') as f:
                        f.write('\n'.join(lines) + '\n')
                    lines = []
                    s = '[%i %s] saved %.2f M lines from %i subreddits, now is %s'%(year, feedback, n/1e6, non_empty_subreddits + 1, sub)
                    print(s)
        if not empty:
            non_empty_subreddits += 1

    with open(path_out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    s = '[%i-%i %s] saved %.2f M lines from %i subreddits'%(year_from, year_to, feedback, n/1e6, non_empty_subreddits )
    with open(path_done, 'w') as f:
        f.write(s)
    print(s)
    return path_out



def split_by_root(path, p_test=0.01):

    print('spliting by root '+path)
    lines = {
        'train': [],
        'vali': [],
        }
    prev = None
    n = 0
    
    for k in lines:
        if len(lines[k]) == 0:
            continue
        open(path + '.' + k, 'w', encoding='utf-8')

    for line in open(path, encoding='utf-8'):
        line = line.strip('\n')
        if not line:
            continue
        cxt = line.split('\t')[3]
        root = cxt.strip().split()[0]
        if root != prev:
            if np.random.random() < p_test:
                k = 'vali'
            else:
                k = 'train'
        #pdb.set_trace()
        lines[k].append(line)
        prev = root
        n += 1
        if n % 1e6 == 0:
            print('read %i M'%(n/1e6))
            for k in lines:
                if len(lines[k]) == 0:
                    continue
                with open(path + '.' + k, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(lines[k]) + '\n')
                lines[k] = []
    
    for k in lines:
        if len(lines[k]) == 0:
            continue
        with open(path + '.' + k, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines[k]))
        lines[k] = []
    

def shuffle(feedback, part, n_temp=10):
    fld = '%s/%s'%(fld_out, feedback)
    path = '%s/raw.tsv.%s'%(fld, part)
    path_out = '%s/%s.tsv'%(fld, part)
    fld_temp = '%s/temp/%s'%(fld_out, feedback)
    
    print('slicing '+path)
    os.makedirs(fld_temp, exist_ok=True)
    lines = [[] for _ in range(n_temp)]

    # split into n_temp files
    for i in range(n_temp):
        open(fld_temp + '/temp%i'%i, 'w', encoding='utf-8') 
    n = 0
    count = [0] * n_temp
    rand = np.random.randint(0, n_temp, 202005)
    for line in open(path, encoding='utf-8'):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        bucket = rand[n % len(rand)]
        lines[bucket].append(line)
        count[bucket] += 1
        n += 1
        if n % 1e6 == 0:
            print('read %i M'%(n/1e6))
            for i in range(n_temp):
                if len(lines[i]) == 0:
                    continue
                with open(fld_temp + '/temp%i'%i, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(lines[i]) + '\n')
                lines[i] = []
    
    for i in range(n_temp):
        with open(fld_temp + '/temp%i'%i, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines[i]))

    # and then merge
    open(path_out, 'w', encoding='utf-8')
    print(fld_temp)
    for i in range(n_temp):
        print('reading temp%i'%i)
        lines = open(fld_temp + '/temp%i'%i, encoding='utf-8').readlines()
        print('shuffling')
        jj = list(range(len(lines)))
        np.random.shuffle(jj)
        print('writing')
        with open(path_out, 'a', encoding='utf-8') as f:
            f.write('\n'.join([lines[j].strip('\n') for j in jj]) + '\n')

def get_subs():
    return ['4chan']
    print('collectiing subs...')
    subs = sorted(os.listdir(fld_subs))
    print('collected %i subs'%len(subs))
    return subs


def build_json(year):
    for date in get_dates(year):
        extract_rc(date)
        extract_rs(date)


def build_basic(year):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    subs = get_subs()
    for sub in subs:
        extract_time(sub, year)
        extract_txt(sub, year, tokenizer)
        extract_trees(sub, year)
        calc_feedback(sub, year, overwrite=False)


def build_pairs(year_from, year_to, feedback):
    subs = get_subs()
    for year in range(year_from, year_to + 1):
        for sub in subs:
            create_pairs(year, sub, feedback, overwrite=False)
            add_seq(sub, year, feedback, overwrite=False)
    path = combine_sub(year_from, year_to, feedback)
    split_by_root(path)
    for part in ['train', 'vali']:
        shuffle(feedback, part)


FLD = 'data'
fld_bz2 = FLD + '/bz2/'
fld_jsonl = FLD + '/jsonl/'
fld_subs = FLD + '/subs/'
fld_out = FLD + '/out/'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('year', type=int)
    parser.add_argument('--year_to', type=int)
    args = parser.parse_args()
    if args.task == 'bz2':
        build_json(args.year)
    elif args.task == 'basic':
        build_basic(args.year)
    elif args.task in ['updown', 'depth', 'width']:
        build_pairs(args.year, args.year_to, args.task)
    else:
        raise ValueError