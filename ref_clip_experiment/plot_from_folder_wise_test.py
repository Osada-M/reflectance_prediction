import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import os

# matplotlib.font_manager._rebuild()

from matplotlib.font_manager import findSystemFonts
fonts = findSystemFonts()
for font in fonts:
    if "times" in font.lower():
        print(font)

# exit()

# font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
# prop = FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'Times New Roman'


## =====================================================================
## 


DIR = os.path.dirname(os.path.abspath(__file__))

LOAD_DIR = f'{DIR}/../results'
SAVE_DIR = f'{DIR}/../plots'

ANSWER_CSV = f'{DIR}/../answers.csv'



## =====================================================================
## 


MASTER_NAMED = 'AR_'


## model, unseen, known
# TARGETS = [
#     ['vgg', 0, 1, 'VGG16'],
#     ['resnet', 0, 1, 'ResNet101'],
#     ['vit', 0, 1, 'ViT-B/32'],
#     ['clip_resnet', 1, 1, 'CLIP (ResNet101)'],
#     ['clip_vit', 1, 1, 'CLIP (ViT-B/32)'],
#     ['gpt-3.5', 1, 0, 'GPT-3.5'],
#     ['gpt-4', 1, 0, 'GPT-4'],
# ]

## target, unseen, known, multi-modal plottable, plot_label
TARGETS = [
    ['vgg',                             1, 1, 0, 'VGG16'],
    ['resnet',                          1, 1, 0, 'ResNet101'],
    ['vit',                             1, 1, 1, 'ViT-B/32'],
    ['clip_resnet',                     1, 1, 0, 'CLIP (ResNet101)'],
    ['clip_vit',                        1, 1, 1, 'CLIP (ViT-B/32, Image-only)'],
    ['clip_vit_multi-modal-add',        0, 0, 1, 'CLIP (ViT-B/32 & , Image + Text, Add)'],
    ['clip_vit_multi-modal-concat',     0, 0, 1, 'CLIP (ViT-B/32, Image + Text, Concat)'],
    ['clip_vit_multi-modal-text-only',  0, 0, 1, 'CLIP (Transformer, Text-only)'],
    ['gpt-3.5',                         1, 0, 0, 'GPT-3.5'],
    ['gpt-4',                           1, 0, 0, 'GPT-4'],
]

ANSWERS = dict()

with open(ANSWER_CSV, 'r') as f:
    lines = f.readlines()

for line in map(str.strip, lines):
    name, ans = line.split(',')
    ANSWERS[name] = float(ans)

ANSWERS = dict(sorted(ANSWERS.items(), key=lambda x: x[1]))


BUMPY = [
    'brush', 'bath_sponge', 'whiteboard_eraser'
]
TRANSPARENT = [
    'coffee', 'green_tea', 'detergent', 'water', 'glass_balls', 'empty_plastic_bottle',
]


FORMAL_NAMES = {
    'monstar' : 'drink 1',
    'monster' : 'drink 1',
    'chipstar' : 'chips 1',
    'chipster' : 'chips 1',
    'pringles' : 'chips 2',
    'cookie' : 'cookies 1',
    'cookiess' : 'cookies',
    'paper_box' : 'tissue_box',
    'pocky' : 'sweets',
    'glass_balls' : 'marbles',
    'zone' : 'drink 2',
    'bond' : 'glue',
    'ritz' : 'crackers',
    # 'water' : 'water bottle',
    'crackers' : 'cookies 2',
    'nori' : 'seaweed',
}


## =====================================================================
## 


# plt.rcParams['font.family'] = 'monospace'




## plot label
## 1st col
# plt.plot([-1, -1], [-1, -1], c='red', label='Ground truth')
# plt.plot([-1, -1], [-1, -1], c='orange', label='Median')

# plt.scatter(-1, -1, marker='^', c=colors['ResNet101(Garcia)'],          label='ResNet101       (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['VGG16(Garcia)'],                label='VGG16             (Mean)')

## 2nd col
# plt.scatter(-1, -1, marker='^', c=colors['GPT-3.5'],            label='GPT-3.5            (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['GPT-4'],              label='GPT-4              (Mean)')

## 3rd col
# plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(1)'],    label='OWL-ViT + FC(1)    (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(32,1)'], label='OWL-ViT + FC(32,1) (Mean)')

# plt.scatter(-1, -1, marker='^', c=colors['CLIP + FC(1)'], label='CLIP + FC(1) (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['CLIP + FC(32,1)'], label='CLIP + FC(32,1) (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['CLIP + GPT-3.5'], label='CLIP + GPT3.5 (Mean)')
# plt.scatter(-1, -1, marker='^', c=colors['CLIP + GPT-4'],   label='CLIP + GPT-4  (Mean)')

# plt.plot([-1, -1], [-1, -1], c='red', label='真値')
# plt.plot([-1, -1], [-1, -1], c='orange', label='中央地')
# plt.scatter(-1, -1, marker='^', c=colors['ResNet101(Garcia)'],  label='ResNet101       (従来手法)')
# plt.scatter(-1, -1, marker='^', c=colors['VGG16(Garcia)'],      label='VGG16           (従来手法)')
# plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(1)'],    label='CLIP + FC(1)')
# plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(32,1)'], label='CLIP + FC(32,1)')


# col = len(TARGETS) + 1


# plt.ylim(0, 1.1)

## =====================================================================
## 


for modal_idx in range(2):
    for is_seen in range(2):
        
        # if not is_seen: continue
        
        for raw in range(3):
            
            # if is_seen and raw: continue
            # if not is_seen and raw == 2: continue
            
            # plt.figure(figsize=(15, 7), dpi=200)
            # plt.figure(figsize=(25, 7), dpi=200)
            plt.figure(figsize=(20, 7), dpi=200)
            
            plt.rcParams['font.size'] = 26

            plot_data = []
            if modal_idx and is_seen:
                plot_data.append([])
            
            plot_keys = []
            x = 1
            
            diff = dict(zip(list(map(lambda x: x[0], TARGETS)), [[] for _ in TARGETS]))
            col = -1

            csv_dir = f'{LOAD_DIR}/{["unseen", "known"][is_seen]}'

            objects = set()
            glaph_table = dict()
            colors = dict()
            markers = dict()

            
            c = iter([
                'black',
                'gray',
                'deepskyblue',
                # 'lightseagreen',
                'lime',
                'gold',
                # 'darkorange',
                'red',
                'purple',
                'blue',
                'green',
                'orange',
                ])
            
            marker = iter([
                '^',
                'o',
                's',
                'D',
                'v',
                '<',
                '>',
                'p',
                'h',
                'x',
            ])
            
            
            omit_number = 0
            is_plot = dict(zip(list(map(lambda x: x[0], TARGETS)), [True for _ in TARGETS]))
            
            for target_vals in TARGETS:
                
                target, is_unseen, is_known, is_multi_modal, plot_label = target_vals
                                
                # if not is_seen:
                #     if not is_unseen:
                #         continue
                # else:
                #     if not is_known:
                #         continue
                
                if modal_idx:
                    is_plot[target] = is_multi_modal
                else:
                    is_plot[target] = (not is_seen and is_unseen) or (is_seen and is_known)
                
                
                # print(target, is_seen, is_unseen, is_known, is_plot[target])
                
                omit_number += not is_plot[target]
                
                csv = f'{csv_dir}/{target}.csv'
                colors[target] = next(c)
                markers[target] = next(marker)
                
                if not os.path.exists(csv):
                    continue
                
                with open(csv, 'r') as f:
                    lines = f.readlines()
                
                for line in map(str.strip, lines):
                    name, *predists = line.split(', ')
                    
                    if not name in glaph_table.keys():
                        glaph_table[name] = dict()

                    glaph_table[name][target] = np.array(list(map(float, predists)))
                    objects.add(name)
                    
                    target_num = len(glaph_table[name]) - omit_number
                    # print(target, target_num, col)
                    
                    if col < target_num:
                        col = len(glaph_table[name])

                ## model names
                if is_plot[target]:
                    print(target)
                    plt.scatter(-1, -1, marker=markers[target], c=colors[target],  label=plot_label, s=200)
                
            # col += 1
            if is_seen:
                col += 2
            else:
                if modal_idx:
                    col += 2
                else:
                    col += 0
            # col += is_seen + 1
            
            ## =====================================================================
            ## 
            
            
            buf = []
            
            if is_seen:
                
                for obj in ANSWERS.keys():
                    if not obj in objects: continue
                    buf.append(obj)
            
            else:
                
                if raw == 0:
                    
                    for obj in ANSWERS.keys():
                        if not obj in objects: continue
                        if obj in BUMPY or obj in TRANSPARENT: continue
                        buf.append(obj)
                    
                elif raw == 1:
                    for obj in ANSWERS.keys():
                        if not obj in objects: continue
                        if obj in BUMPY:
                            buf.append(obj)
                            
                elif raw == 2:
                    for obj in ANSWERS.keys():
                        if not obj in objects: continue
                        if obj in TRANSPARENT:
                            buf.append(obj)
                        
            
            
            del objects
            objects = buf.copy()
            del buf
            
            
            ## =====================================================================
            ## 
            
            
            ## plot each results
            for i, obj in enumerate(objects):
            
                plot_keys.append(obj)
                if modal_idx and is_seen:
                    x += 1
                    plt.plot([x-1, x+col-3], [ANSWERS[obj], ANSWERS[obj]], c='red', zorder=1)
                else:
                    plt.plot([x-0.5, x+col-2.5], [ANSWERS[obj], ANSWERS[obj]], c='red', zorder=1)
                
                idx = 0
                for model in TARGETS:
                    
                    model, *_ = model
                    
                    if not model in glaph_table[obj].keys(): continue
                    
                    data = glaph_table[obj][model]
                    if not len(data): continue
                    
                    if is_plot[model]:
                        plt.scatter([x+idx], np.mean(data), c=colors[model], marker=markers[model], zorder=3, s=150)
                        plot_data.append(data)
                        idx += 1
                        # print(model)
                        
                    diff[model] += list(np.array(data) - ANSWERS[obj])
                
                # if is_plot[model]:
                plot_data.append([])
                plot_data.append([])
                
                if modal_idx and is_seen:
                    plot_data.append([])
                    x -=1
                        
                x += col

            ## object names
            for p, plot in enumerate(plot_keys):
                
                print(plot)
                
                for before, after in FORMAL_NAMES.items():
                    plot = plot.replace(before, after)
                
                plot = plot[0].upper() + plot[1:]
                
                underber = plot.split('_')
                if len(underber) > 1:
                    
                    if is_seen:
                        plot = ' '.join(underber[:-1]) + f' ({underber[-1]})'
                    else:    
                        plot = plot.replace('_', ' ')
                    
                    ## others
                    plot = plot.replace('Whiteboard', 'Board')
                    plot = plot.replace('Empty plastic', 'Empty')
                
                plot_keys[p] = plot
            
            # plt.title('reflectance')
            
            plt.boxplot(plot_data, whis=[0,100])
            plt.xlim(0, len(plot_keys)*col + 1)
            
            # plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.23)
            # plt.xticks(np.arange(1, len(plot_keys)+1)*col -(col/2), plot_keys, rotation=33)
            plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.18)
                
            # plt.xlabel('Objects')
            plt.ylabel('Reflectance')
            
            ## =====================================================================
            ## ncol !!!!!!!!!!!!!!! here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # plt.ylim(-0.01, 1.1)
            plt.ylim(-0.01, 1.3)
            if is_seen:
                # plt.ylim(0.33, 1.05)
                plt.legend(loc='upper left', ncol=2)
                plt.xticks(np.arange(1, len(plot_keys)+1)*col -(col/2), plot_keys, rotation=20)
            else:
                if modal_idx:
                    plt.ylim(-0.01, 1.5)
                    plt.legend(loc='upper left', ncol=2)
                else:
                    plt.legend(loc='upper left', ncol=4)
                plt.xticks(np.arange(1, len(plot_keys)+1)*col -(col/2), plot_keys, rotation=0)
            
            ## =====================================================================

            ## remove line top and right
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            plt.savefig(f'{SAVE_DIR}/{["unseen", "known"][is_seen]}{["", "_multi_modal"][modal_idx]}_{raw}.png')    
            
            plt.close()


            mean_error = dict(zip(list(map(lambda x: x[0], TARGETS)), [[] for _ in TARGETS]))
            std_var = dict(zip(list(map(lambda x: x[0], TARGETS)), [[] for _ in TARGETS]))

            for model in TARGETS:
                
                model, *_ = model
                
                vals = diff[model]
                if not len(vals): continue
                mean_error[model] += list(map(abs, vals))
                std_var[model] = np.std(vals)
                

            insert_space = lambda *x: f'{x[0]}{" "*(x[1]-len(x[0]))}'

            with open(f'{SAVE_DIR}/{["unseen", "known"][is_seen]}_{raw}.txt', mode='w') as f:
                
                print(f'{insert_space("Model", 30)}{insert_space("Mean Error", 30)}{insert_space("Standard Variance", 30)}{insert_space("LaTeX style", 30)}', file=f)
                print(f'{"-" * (30 + 30 + 30 + 30)}', file=f)
                
                for model in TARGETS:
                    
                    model, *_ = model
                    
                    vals = diff[model]
                    if not len(vals): continue
                    mean_error[model] = np.mean(mean_error[model])
                    
                    table_view = insert_space(f'$ {mean_error[model]:.03f} \pm {std_var[model]:.03f} $', 30)
                    print(f'{insert_space(model, 30)}{insert_space(str(mean_error[model]), 30)}{insert_space(str(std_var[model]), 30)}{table_view}', file=f)

                print(mean_error)
                print(std_var)
            
