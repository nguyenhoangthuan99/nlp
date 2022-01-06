import os
import re
import shutil
data_dir = 'data/viki2021'
out_dir = 'data/viki2021_processed'
files = os.listdir(data_dir)

if os.path.exists(out_dir) is False:
    os.mkdir(out_dir)
else:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

for f in files:
    in_file = data_dir + '/' + f
    out_file = out_dir + '/' + f

    out_pf = open(out_file, 'w')
    with open(in_file, 'r') as lines:
        for line in lines:
            line = re.sub('[,:;\.]+ ', ' ', line)
            line = re.sub('[,:;\.]+\n', ' ', line)
            line = re.sub('[ ]+', ' ', line)
            if len(line.split(' ')) < 2:
                continue
            out_pf.write(line.lower())
    out_pf.close()

