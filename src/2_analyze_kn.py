import json
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter
import random
import sys
import seaborn as sns
import pandas as pd
from pandas.core.frame import DataFrame

kn_dir = '../results/bert-large-cased-req-compl/kn/'
fig_dir = '../results/bert-large-cased-req-compl/figs/'

class Tee(object):
    def __init__(self, filename, mode="a"):
        self.file = open(os.path.join(kn_dir, filename), mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
sys.stdout = Tee("analyzed_kn.txt", "a")

# =========== stat kn_bag ig ==============

y_points = []   
tot_bag_num = 0
tot_rel_num = 0
tot_kneurons = 0
kn_bag_counter = Counter()
for filename in os.listdir(kn_dir):
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_bag_counter.update([kn[0]])
                y_points.append(kn[0])
        tot_bag_num += len(kn_bag_list)
for k, v in kn_bag_counter.items():
    tot_kneurons += kn_bag_counter[k]
for k, v in kn_bag_counter.items():
    kn_bag_counter[k] /= tot_kneurons

# average # Kneurons
print('average ig_kn', tot_kneurons / tot_bag_num)

# =========== stat kn_bag base ==============

tot_bag_num = 0
tot_rel_num = 0
tot_kneurons = 0
base_kn_bag_counter = Counter()
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                base_kn_bag_counter.update([kn[0]])
        tot_bag_num += len(kn_bag_list)
for k, v in base_kn_bag_counter.items():
    tot_kneurons += base_kn_bag_counter[k]
for k, v in base_kn_bag_counter.items():
    base_kn_bag_counter[k] /= tot_kneurons
# average # Kneurons
print('average base_kn', tot_kneurons / tot_bag_num)

# =========== plot knowledge neuron distribution ===========
# Find the maximum layer index among all kns
max_layer = max(kn_bag_counter.keys()) if kn_bag_counter else 0

# Dynamically set figure width: 0.5 inch per layer, minimum 8 inches
fig_width = max(8, (max_layer + 1) * 0.5)
plt.figure(figsize=(fig_width, 4))  # Slightly taller for more readable y-axis

x = np.array([i + 1 for i in range(max_layer + 1)])
y = np.array([kn_bag_counter.get(i, 0) for i in range(max_layer + 1)])
plt.xlabel('Layer', fontsize=18)
plt.ylabel('Percentage', fontsize=18)
plt.xticks(x, labels=x, fontsize=14)

# Set y-ticks dynamically based on data range
y_min = min(-0.4, -y.max() - 0.03)
y_max = max(0.5, y.max() + 0.03)
yticks = np.linspace(y_min, y_max, num=10)
plt.yticks(yticks, labels=[f'{int(val*100)}%' for val in yticks], fontsize=14)

plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, labelsize=14)
plt.ylim(y_min, y_max)
plt.xlim(0.3, max_layer + 1 + 0.7)
bottom = -y
y = y * 2
plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, 'kneurons_distribution.pdf'), dpi=100)
plt.close()

# ========================================================================================
#                       knowledge neuron intersection analysis
# ========================================================================================


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def cal_intersec(kn_bag_1, kn_bag_2):
    kn_bag_1 = set(['@'.join(map(str, kn)) for kn in kn_bag_1])
    kn_bag_2 = set(['@'.join(map(str, kn)) for kn in kn_bag_2])
    return len(kn_bag_1.intersection(kn_bag_2))

# ====== load ig kn =======

kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir):
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list

# ig inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i]
            kn_bag_2 = kn_bag_list[j]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'ig kn has on average {inner_ave_intersec} inner kn interseciton')

# ig inter
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(len_kn_bag_list):
        for j in range(100):
            kn_bag_1 = kn_bag_list[i]
            valid_other_rels = [
                x for x in kn_bag_list_per_rel.keys()
                if x != rel and len(kn_bag_list_per_rel[x]) > 0
            ]
            if not valid_other_rels:
                continue  # skip this round if no valid other relations
            other_rel = random.choice(valid_other_rels)
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
            
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inter_ave_intersec.append(num_intersec)

inter_ave_intersec = np.array(inter_ave_intersec).mean()
print(f'ig kn has on average {inter_ave_intersec} inter kn interseciton')


# ====== load base kn =======
kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list

# base inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i]
            kn_bag_2 = kn_bag_list[j]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'base kn has on average {inner_ave_intersec} inner kn interseciton')

# base inter
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)

    for i in range(len_kn_bag_list):
        for j in range(100):
            kn_bag_1 = kn_bag_list[i]
            # Filter out empty or same-relation bags

            valid_other_rels = [
                x for x in kn_bag_list_per_rel.keys()
                if x != rel and len(kn_bag_list_per_rel[x]) > 0
            ]

            if not valid_other_rels:
                continue  # skip this round if there's no valid relation to compare to
            other_rel = random.choice(valid_other_rels)
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]

            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)

            inter_ave_intersec.append(num_intersec)

if inter_ave_intersec:
    inter_ave_intersec = np.array(inter_ave_intersec).mean()
else:
    inter_ave_intersec = 0  # or np.nan, depending on your use case
print(f'base kn has on average {inter_ave_intersec} inter kn interseciton')

