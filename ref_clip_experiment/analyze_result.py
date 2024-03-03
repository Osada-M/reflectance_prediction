import numpy as np
import matplotlib.pyplot as plt


KEY_REP = {
    'sauce_brown'       : 'sauce',
    'monster_pink'      : 'monster',
    'chipstar_orange'   : 'chipstar',
    'tea_yellow'        : 'tea_pet',
    'pocky_yellow'      : 'pocky',
    'jelly_purple'      : 'qoo_jelly',
    'pringles_green'    : 'pringles',
    'yogurt_pink'       : 'yogurt',
    'paperbox_blue'     : 'tissue',
}

def key_rep(key):
    
    if key in KEY_REP.keys():
        return KEY_REP[key]
    return key


def call(dir, model_id, is_train):
    
    
    with open(f'{dir}/results/{model_id}/result_detail{["", "_train"][is_train]}.txt', 'r') as f:
        lines = f.readlines()

    header = lines.pop(0)
    keys = header.split(':')[1].split()

    data_fromtxt = {}
    for line in lines:
        line = line.replace('[', '').replace(']', '')
        key, vals = line.split(':')
        key = key.replace(' ', '')
        vals = np.array(list(map(float, vals.split())))
        
        data_fromtxt[key] = vals

    data = dict(zip(keys, [{'ans':0, 'mean':[], 'max':0, 'min':1<<10, 'std':[]} for _ in keys]))

    for i, (key, vals) in enumerate(data_fromtxt.items()):
        
            
        for j, (name, val) in enumerate(zip(keys, vals)):
            
            if key == 'ans':
                data[name]['ans'] = val
            
            elif key == 'pred':
                
                data[name]['max'] = max(data[name]['max'], val)
                data[name]['min'] = min(data[name]['min'], val)
                data[name]['mean'].append(val)
                data[name]['std'].append(val)


    with open(f'{dir}/results/{model_id}/result_for_plot{["", "_train"][is_train]}.txt', 'w') as f:
        for key, val in data.items():
            print(f'{key_rep(key)} : {val["mean"]}', file=f)

    for name in set(keys):
        data[name]['mean'] = np.mean(data[name]['mean'])
        data[name]['std'] = np.std(data[name]['std'])
            
    with open(f'{dir}/results/{model_id}/result_aggregation{["", "_train"][is_train]}.txt', 'w') as f:
        for key, val in data.items():
            print(f'{key} : {val}', file=f)


    loss = []
    previous_model = model_id
    
    while previous_model != 'none':
        
        print(f'read loss : {previous_model}')
        
        with open(f'{dir}/results/{previous_model}/history.txt', 'r') as f:
            lines = f.readlines()[0]
            loss_ = lines.split(']')[0].split('[')[1].replace(' ', '')
            loss_ = list(map(float, (loss_).split(',')))
        
        loss = loss_ + loss
        
        with open(f'{dir}/results/{previous_model}/config.txt', 'r') as f:
        
            lines = f.readlines()
            data = [elem for elem in map(
                lambda x: list(map(lambda y: y.replace(' ', '').replace('\n', ''), x.split(':'))), lines
                )]
            data = dict(data)
            previous_model = data['load_path']
        
    plt.figure(figsize=(15, 10), dpi=100)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.plot(range(len(loss)), loss)
    
    plt.savefig(f'{dir}/results/{model_id}/history.png')
    
    
if __name__ == '__main__':
    
    import os
    
    DIR = os.path.dirname(os.path.abspath(__file__))
    NAME = '20230624_1817_33'
    
    call(DIR, NAME)