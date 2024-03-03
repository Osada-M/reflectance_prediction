import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import os

font_path = "/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf"
font_prop = FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()


DIR = os.path.dirname(os.path.abspath(__file__))


# MASTER_NAMED = None
MASTER_NAMED = 'RSJ2023_1'


KEYS = ['monster', 'chipstar', 'tea_pet', 'qoo_jelly', 'pringles', 'pocky', 'tissue']

TARGETS = [
    'ResNet101(Garcia)',
    'VGG16(Garcia)',
    # 'ResNet101',
    # 'VGG16',
    # 'GPT-3.5',
    # 'GPT-4',
    'OWL-ViT + FC(1)',
    'OWL-ViT + FC(32,1)',
    # 'CLIP + GPT-3.5',
    # 'CLIP + GPT-4',
    # 'CLIP + FC(1)',
    # 'CLIP + FC(32,1)',
]

LABELS = {
    'sauce' : 0.36375,
    'monster' : 0.411,
    'chipstar' : 0.43925,
    'tea_pet' : 0.55925,    ## 紅茶花伝
    'pocky' : 0.7285,       ## ポッキー（.495, .625, .850, .944）
    'qoo_jelly' : 0.589,    ## Qooゼリー
    'pringles' : 0.684,
    'yogurt' : 0.7495,      ## ブルガリア
    'tissue' : 0.88975,   ## ティッシュ
}

DATA = dict(zip(KEYS, [dict() for _ in KEYS]))
for key in KEYS:
    DATA[key] = dict(zip(TARGETS, [None]*len(TARGETS)))
    DATA[key]['Ground truth'] = LABELS[key]


DATA['tea_pet']['GPT-4'] =          [0.58, 0.25, 0.45, 0.55, 0.55, 0.55, 0.5, 0.38, ]
DATA['tea_pet']['GPT-3.5'] =        [0.6, 0.825, 0.9, 0.85, 0.85, 0.6, 0.775, 0.85, ]
DATA['tea_pet']['CLIP + GPT-4'] =   [0.6, 0.8, 0.82, 0.6, ]
DATA['tea_pet']['CLIP + GPT-3.5'] = [0.9, 0.9, 0.87, 0.75, 0.8, 0.7, 0.4, 0.875, ]
DATA['tea_pet']['ResNet101(Garcia)'] =      [0.616, 0.616, 0.616, 0.616, 0.616, 0.616, ]
DATA['tea_pet']['VGG16(Garcia)'] =            [0.792, 0.547, 0.547, 0.547, 0.547, 0.547, ]
DATA['tea_pet']['OWL-ViT + FC(1)'] = [0.48667213, 0.51702124, 0.51022965, 0.5231254, 0.48194116, 0.5037775]
DATA['tea_pet']['OWL-ViT + FC(32,1)'] = [0.5331146, 0.53341186, 0.56140345, 0.5687662, 0.547324, 0.554208]
DATA['tea_pet']['CLIP + FC(1)'] = [0.6061031, 0.6127574, 0.6316883, 0.6212924, 0.5829278, 0.6177579]
DATA['tea_pet']['CLIP + FC(32,1)'] = [0.6050006, 0.57197964, 0.54898745, 0.52764267, 0.50741726, 0.5483865]
DATA['tea_pet']['ResNet101'] = []
DATA['tea_pet']['VGG16'] = []

DATA['qoo_jelly']['GPT-4'] =        [0.48, 0.28, 0.42, 0.5, 0.5, 0.50, 0.45, 0.39, ]
DATA['qoo_jelly']['GPT-3.5'] =      [0.6, 0.725, 0.65, 0.65, 0.85, 0.1, 0.675, 0.85, ]
DATA['qoo_jelly']['CLIP + GPT-4'] = [0.7, 0.55, 0.78, 0.35, 0.55, 0.55, 0.55, 0.5, ]
DATA['qoo_jelly']['CLIP + GPT-3.5'] = [0.1, 0.2, 0.05, 0.125, 0.2, 0.25, 0.5, 0.35, ]
DATA['qoo_jelly']['ResNet101(Garcia)'] =    [0.546, 0.546, 0.546, 0.546, 0.546, 0.546, ]
DATA['qoo_jelly']['VGG16(Garcia)'] =          [0.399, 0.399, 0.399, 0.588, 0.595, 0.399, ]
DATA['qoo_jelly']['OWL-ViT + FC(1)'] = [0.5524756, 0.52372164, 0.48973283, 0.50637996, 0.54550034, 0.5256608]
DATA['qoo_jelly']['OWL-ViT + FC(32,1)'] = [0.60401434, 0.5731531, 0.5403561, 0.5544323, 0.60703474, 0.5911666]
DATA['qoo_jelly']['CLIP + FC(1)'] = [0.6415884, 0.62675774, 0.5853772, 0.67049986, 0.625629, 0.6028925]
DATA['qoo_jelly']['CLIP + FC(32,1)'] = [0.50955814, 0.54064524, 0.534525, 0.63092446, 0.60103613, 0.56016266]
DATA['qoo_jelly']['ResNet101'] = []
DATA['qoo_jelly']['VGG16'] = []

DATA['pringles']['GPT-4'] =         [0.52, 0.55, 0.53, 0.45, 0.7, 0.78, 0.65, 0.55, ]
DATA['pringles']['GPT-3.5'] =       [0.75, 0.55, 0.684, 0.7, 0.8, 0.725, 0.5, 0.45, ]
DATA['pringles']['CLIP + GPT-4'] =  [0.7, 0.65, 0.35, 0.7, 0.55, 0.5, ]
DATA['pringles']['CLIP + GPT-3.5'] = [0.1, 0.4, 0.15, 0.125, 0.6, 0.4, 0.65, 0.55, ]
DATA['pringles']['ResNet101(Garcia)'] =     [0.662, 0.662, 0.662, 0.662, 0.662, 0.662, ]
DATA['pringles']['VGG16(Garcia)'] =           [0.399, 0.399, 0.662, 0.662, 0.662, 0.399, ]
DATA['pringles']['OWL-ViT + FC(1)'] = [0.57803875, 0.61285686, 0.6203909, 0.6100888, 0.60444355, 0.5720602]
DATA['pringles']['OWL-ViT + FC(32,1)'] = [0.5970948, 0.6545352, 0.66252613, 0.65245754, 0.66977286, 0.5964456]
DATA['pringles']['CLIP + FC(1)'] = [0.65273154, 0.6173087, 0.6537869, 0.60561925, 0.6204152, 0.65336907]
DATA['pringles']['CLIP + FC(32,1)'] = [0.6908058, 0.6504972, 0.6972356, 0.6566547, 0.6501522, 0.6814018]
DATA['pringles']['ResNet101'] = []
DATA['pringles']['VGG16'] = []

DATA['pocky']['GPT-4'] =            [0.38, 0.15, 0.15, 0.15, 0.65, 0.4, 0.25, 0.52, ]
DATA['pocky']['GPT-3.5'] =          [0.6, 0.25, 0.9, 0.06, 0.6, 0.5, 0.3, 0.6, ]
DATA['pocky']['CLIP + GPT-4'] =     [0.68, 0.7, 0.55, 0.5, ]
DATA['pocky']['CLIP + GPT-3.5'] =   [0.1, 0.5, 0.2, 0.87, 0.10, 0.6, 0.7, 0.4, ]
DATA['pocky']['ResNet101(Garcia)'] =        [0.944, 0.944, 0.944, 0.944, 0.944, 0.944, ]
DATA['pocky']['VGG16(Garcia)'] =              [0.860, 0.860, 0.860, 0.547, 0.792, 0.850, ]
DATA['pocky']['OWL-ViT + FC(1)'] =  [0.7469574, 0.7677029, 0.7222686, 0.7694543, 0.77329254, 0.74159676]
DATA['pocky']['OWL-ViT + FC(32,1)'] = [0.78600746, 0.8401232, 0.70674336, 0.8099172, 0.82193696, 0.78834456]
DATA['pocky']['CLIP + FC(1)'] = [0.65273154, 0.6173087, 0.6537869, 0.60561925, 0.6204152, 0.65336907]
DATA['pocky']['CLIP + FC(32,1)'] = [0.91084677, 0.94169027, 0.9091126, 0.66365623, 0.94686514, 0.8355267]
DATA['pocky']['ResNet101'] = []
DATA['pocky']['VGG16'] = []

DATA['tissue']['GPT-4'] =           [0.62, 0.70, 0.67, 0.65, 0.7, 0.684, 0.675, 0.68, ]
DATA['tissue']['GPT-3.5'] =         [0.6, 0.675, 0.45, 0.04, 0.65, 0.3, 0.684, 0.5, ]
DATA['tissue']['CLIP + GPT-4'] =    [0.684, 0.6, 0.65, 0.68, 0.68, 0.68, 0.67, 0.67, ]
DATA['tissue']['CLIP + GPT-3.5'] =  [0.25, 0.15, 0.85, 0.055, 0.65, 0.075, 0.9, 0.15, ]
DATA['tissue']['ResNet101(Garcia)'] =       [0.830, 0.830, 0.830, 0.830, 0.830, 0.830, ]
DATA['tissue']['VGG16(Garcia)'] =             [0.860, 0.860, 0.860, 0.860, 0.860, 0.860, ]
DATA['tissue']['OWL-ViT + FC(1)'] = [0.8736549, 0.8775271, 0.8888131, 0.8886628, 0.8767268, 0.88868165]
DATA['tissue']['OWL-ViT + FC(32,1)'] = [0.8623712, 0.88853955, 0.8940728, 0.8883741, 0.89138997, 0.88372135]
DATA['tissue']['CLIP + FC(1)'] = [0.7168146, 0.6880641, 0.7222372, 0.62322384, 0.63949734, 0.6868175]
DATA['tissue']['CLIP + FC(32,1)'] = [0.82811606, 0.7430019, 0.8284758, 0.711708, 0.83224356, 0.8318058]
DATA['tissue']['ResNet101'] = []
DATA['tissue']['VGG16'] = []


# for key in DATA.keys():
#     if DATA[key]['GPT-3.5'] is None: continue
#     DATA[key]['GPT-3.5'] += DATA[key]['CLIP + GPT-3.5']
#     DATA[key]['GPT-4'] += DATA[key]['CLIP + GPT-4']

# DATA['monster']['GPT-4'] = [    0.411, ]

# DATA['chipstar']['GPT-4'] = []


# plt.rcParams['font.family'] = 'monospace'
plt.figure(figsize=(15, 8), dpi=200)

plot_data = []
plot_keys = []
x = 1

c = iter([
    'black',
    'gray',
    'deepskyblue',
    'lightseagreen',
    'lime',
    'gold',
    'darkorange',
    'red',
    ])

colors = {
    'ResNet101(Garcia)': next(c),
    'VGG16(Garcia)': next(c),
    
    
    # 'GPT-3.5': next(c),
    # 'GPT-4': next(c),
    
    'OWL-ViT + FC(1)': next(c),
    'OWL-ViT + FC(32,1)': next(c),
    
    # 'CLIP + FC(1)': next(c),
    # 'CLIP + FC(32,1)': next(c),
    
    # 'CLIP + GPT-3.5': next(c),
    # 'CLIP + GPT-4': next(c),
}

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

plt.plot([-1, -1], [-1, -1], c='red', label='真値')
plt.plot([-1, -1], [-1, -1], c='orange', label='中央地')
plt.scatter(-1, -1, marker='^', c=colors['ResNet101(Garcia)'],  label='ResNet101       (従来手法)')
plt.scatter(-1, -1, marker='^', c=colors['VGG16(Garcia)'],      label='VGG16           (従来手法)')
plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(1)'],    label='CLIP + FC(1)')
plt.scatter(-1, -1, marker='^', c=colors['OWL-ViT + FC(32,1)'], label='CLIP + FC(32,1)')


col = len(TARGETS) + 1
diff = dict(zip(TARGETS, [[] for _ in TARGETS]))

for key in KEYS:
    # if DATA[key]['OWL-ViT + FC(1)'] is None: continue
    if DATA[key]['VGG16(Garcia)'] is None: continue
    
    plot_keys.append(key)
    
    plt.plot([x-0.5, x+col-1.5], [DATA[key]['Ground truth'], DATA[key]['Ground truth']], c='red', zorder=1)
    
    idx = 0
    for model in TARGETS:
        data = DATA[key][model]
        if not len(data): continue
        plt.scatter([x+idx], np.mean(data), c=colors[model], marker='^', zorder=3)
        plot_data.append(data)
        diff[model] += list(np.array(data) - DATA[key]['Ground truth'])
        idx += 1
    
    plot_data.append([])
    x += col
    
plt.title('reflectance')

plt.boxplot(plot_data, showmeans=True, whis=[0,100])
plt.xlim(0, len(plot_keys)*col + 1)
plt.ylim(0, 1.1)
plt.xticks(np.arange(1, len(plot_keys)+1)*col -(col/2), plot_keys)

plt.legend(loc='upper left', ncol=2) ## ncol !!!!!!! here !!!!!! ==============================================

plt.savefig(f'{DIR}/plots/reflectance_{"_".join(map(lambda x: x.replace(" ", ""), TARGETS)) if MASTER_NAMED is None else MASTER_NAMED}.png')    


mean_error = dict(zip(TARGETS, [[] for _ in TARGETS]))
std_var = dict(zip(TARGETS, [[] for _ in TARGETS]))

for model in TARGETS:
    vals = diff[model]
    if not len(vals): continue
    mean_error[model] += list(map(abs, vals))
    std_var[model] = np.std(vals)
     

insert_space = lambda *x: f'{x[0]}{" "*(x[1]-len(x[0]))}'

with open(f'{DIR}/plots/reflectance_{"_".join(map(lambda x: x.replace(" ", ""), TARGETS))}.txt', mode='w') as f:
    
    print(f'{insert_space("Model", 20)}{insert_space("Mean Error", 30)}{insert_space("Standard Variance", 30)}', file=f)
    print(f'{"-" * (20 + 30 + 30)}', file=f)
    
    for model in TARGETS:
        vals = diff[model]
        if not len(vals): continue
        mean_error[model] = np.mean(mean_error[model])
        
        print(f'{insert_space(model, 20)}{insert_space(str(mean_error[model]), 30)}{insert_space(str(std_var[model]), 30)}', file=f)

    print(mean_error)
    print(std_var)
    
