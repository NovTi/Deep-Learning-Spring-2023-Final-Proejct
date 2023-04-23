
import os
import sys

# SEP_INTERVAL = 4

parse_metric = lambda x: x.strip().split()[-1]
parse_pretrain_eps = lambda x: list(map(parse_metric, [x[6], x[3], x[4]]))
parse_pretrain_std = lambda x: list(map(parse_metric, [x[7], x[4], x[5]]))
pad_string = lambda x, l: [i.ljust(j) for i, j in zip(x, l)]

dump = []
root = f'{sys.argv[1]}'
column_size = [12, 40, 10, 15]
column_name = ['slurm_id', 'exp_id', 'epoch', 'loss']

def get_file_name(path):
    f_list = os.listdir(path)
    files = []
    for i in f_list:
        if i[-4:] == '.out':
            files.append(i)
    return files

print(f'\ndumping results under {root}:\n')
print(' '.join(pad_string(column_name, column_size)))

file_lst = get_file_name(root)
for i, d in enumerate(sorted(file_lst)):
    path = os.path.join(root, d)
    slurm_id = path[8:-4]
    with open(path, 'r') as f:
        lines = []
        exp_id = None
        for l in f.read().split('\n'):
            # deal with exp_id
            if l[:8] == '[exp_id]':
                exp_id = l[10:]
            # deal with Epoch Results
            if 'Training time' in l:
                lines.append(l)
        # lines = [l for l in f.read().split('\n') if 'Training time' in l]

    if len(lines) == 0 or exp_id is None:
        results = [slurm_id, 'None', 'None', 'None']
    else:
        info = lines[-1].split(' ')
        epoch = info[info.index('Epoch')+1]
        loss = '{} {}'.format(info[-2], info[-1])
        
        results = [slurm_id, exp_id, epoch, loss]
    # print results
    print(' '.join(pad_string(results, column_size)))
    



# for i, d in enumerate(sorted(os.listdir(root))):
#     if i % SEP_INTERVAL == 0:
#         print()
#     path = os.path.join(root, d, 'output.log')
#     with open(path, 'r') as f:
#         lines = [l for l in f.read().split('\n') if l.startswith('Testing')]
#     if len(lines) >= 4:
#         # func = parse_pretrain_eps if d.find('eps') != -1 else parse_pretrain_std
#         func = parse_pretrain_eps
#         dump.append([d] + func(lines[-1].split('|')) + [str(len(lines))])
#     else:
#         dump.append([d, 'TBD', 'TBD', 'TBD', str(len(lines))])
#     print(' '.join(pad_string(dump[-1], column_size)))
# print()

# with open(f"./csv/{sys.argv[1]}.csv", "w") as f:
#     f.writelines([f"{','.join(l)}\n" for l in dump])
        