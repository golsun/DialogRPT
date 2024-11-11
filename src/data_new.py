"""
data source:
https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10
https://academictorrents.com/details/20520c420c6c846f555523babc8c059e9daa8fc5
"""

def zst2jsonl(path_zst):
    path_jsonl = path_zst + '.jsonl'
    open(path_jsonl, 'w')
    n_line = 0
    out = []
    with open(path_zst, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                chunk = reader.read(2**24)  # 16mb chunks
                if not chunk:
                    break

                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    object = json.loads(line)
                    out.append(json.dumps(object, ensure_ascii=False))
                    n_line += 1
                    if n_line % 1e5 == 0:
                        print(n_line)
                        with open(path_zst + '.jsonl', 'a') as f:
                            f.write('\n'.join(out) + '\n')
                        out = []
                previous_line = lines[-1]

    if out:
        with open(path_zst + '.jsonl', 'a') as f:
            f.write('\n'.join(out) + '\n')
    print(n_line)
