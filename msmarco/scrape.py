import sys
import pandas as pd
import html2text
import sh
import urllib.request
import io
import os
import time
import ssl
import multiprocessing as mp
import json
import socket

SUPPORTED_ENCODINGS = ['utf-8', 'latin1']
USER_AGENT = 'Mozilla/5.0 (iPad; U; CPU OS 3_2_1 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Mobile/7B405'

data = pd.read_json(sys.argv[1])
directory = os.path.join('full', sys.argv[1])

sh.mkdir('-p', directory)

output_buffer = {}
count = 0
gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

def find_passages(pair):
    index, row = pair
    print(index, os.getpid(), 'start')
    passages = row.passages
    passage_set = []

    for i, item in enumerate(passages):
        parser = html2text.HTML2Text()
        parser.ignore_links = True
        parser.ignore_emphasis = True
        parser.ignore_images = True
        parser.ignore_tables = True

        print(index, os.getpid(), i, 'start')
        full_passages = []
        url = item['url']
        r = urllib.request.Request(
                url,
                headers={
                    'User-Agent': USER_AGENT,
                    'Accept': 'text/html; charset=utf-8',
                    'Accept-Charset': 'UTF-8'
                    }
                )

        success = False

        for ssl_context in [None, gcontext]:
            try:
                with urllib.request.urlopen(r, context=ssl_context, timeout=30) as f:
                    buf = io.BytesIO()
                    while True:
                        bytes_read = buf.write(f.read(2 ** 20))
                        if bytes_read == 0:
                            break
                        print(index, os.getpid(), i, buf.tell(), 'bytes read')
                    content = buf.getvalue()
                    for encoding in SUPPORTED_ENCODINGS:
                        try:
                            content_string = content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            time.sleep(1)
                            continue
                    else:
                        raise UnicodeDecodeError('All supported encodings tried')
                    text = parser.handle(content_string)
                    success = True
                break
            except socket.timeout:
                print(index, os.getpid(), i, 'Timeout', url)
                time.sleep(1)
                break
            except urllib.error.HTTPError as e:
                print(index, os.getpid(), i, 'HTTP Error', e.getcode(), url)
                time.sleep(1)
                break
            except Exception as e:
                print(index, os.getpid(), i, sys.exc_info(), url)
                time.sleep(1)
                break

        if success:
            buf = io.StringIO(text)
            current_passage = []
            for line in buf:
                line_stripped = line.strip()
                if len(line_stripped) > 0:
                    current_passage.append(line_stripped)
                else:
                    current_passage_text = ' '.join(current_passage).strip()
                    current_passage_tokens = current_passage_text.split()
                    if len(current_passage_tokens) > 10:
                        full_passages.append(current_passage_text)
                    current_passage.clear()

            passage_set.append(full_passages)
        else:
            passage_set.append('')

    print(index, os.getpid(), 'done')
    return index, passage_set

n_rows = data.shape[0]
chunk_size = 500
for i in range(count * chunk_size, n_rows, chunk_size):
    data_chunk = data.iloc[i:i+chunk_size]
    with mp.Pool(mp.cpu_count()) as pool:
        output_buffer = dict(pool.map(find_passages, data_chunk.iterrows()))
    #output_buffer = dict([find_passages((str(data.iloc[i].name), data.iloc[i]))])
    destination = os.path.join(directory, '%d.json' % count)
    with open(destination, 'w') as f:
        json.dump(output_buffer, f)
    output_buffer = {}
    print('flushed to %s' % destination)
    count += 1
