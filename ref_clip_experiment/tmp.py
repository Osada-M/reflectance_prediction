import os

DIR = os.path.dirname(os.path.abspath(__file__))

load = f'{DIR}/data/train.txt'
save = f'{DIR}/tmp.txt'

with open(load, 'r') as f:
    lines = f.readlines()


with open(save, 'w') as f:

    old_name = ''
    
    for line in map(str.strip, lines):
        
        path, ans = line.split(' ')
        object_name = '_'.join(path.split('/')[-1].replace('.png', '').split('_')[:-1])
        
        if old_name != object_name:
            
            old_name = object_name
            print(f'{object_name}, {ans}', file=f)
    