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

def analyze_kn_dir(kn_dir, fig_dir):
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

    print('average ig_kn', tot_kneurons / tot_bag_num if tot_bag_num else 0)

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
    print('average base_kn', tot_kneurons / tot_bag_num if tot_bag_num else 0)

    # =========== plot knowledge neuron distribution ===========
    max_layer = max(kn_bag_counter.keys()) if kn_bag_counter else 0
    fig_width = max(8, (max_layer + 1) * 0.5)
    plt.figure(figsize=(fig_width, 4))
    x = np.array([i + 1 for i in range(max_layer + 1)])
    y = np.array([kn_bag_counter.get(i, 0) for i in range(max_layer + 1)])
    plt.xlabel('Layer', fontsize=18)
    plt.ylabel('Percentage', fontsize=18)
    plt.xticks(x, labels=x, fontsize=14)
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
    os.makedirs(fig_dir, exist_ok=True)
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
    inner_ave_intersec = np.array(inner_ave_intersec).mean() if inner_ave_intersec else 0
    print(f'ig kn has on average {inner_ave_intersec} inner kn interseciton')

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
                    continue
                other_rel = random.choice(valid_other_rels)
                other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
                kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
                num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
                inter_ave_intersec.append(num_intersec)
    inter_ave_intersec = np.array(inter_ave_intersec).mean() if inter_ave_intersec else 0
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
    inner_ave_intersec = np.array(inner_ave_intersec).mean() if inner_ave_intersec else 0
    print(f'base kn has on average {inner_ave_intersec} inner kn interseciton')

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
                    continue
                other_rel = random.choice(valid_other_rels)
                other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
                kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
                num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
                inter_ave_intersec.append(num_intersec)
    inter_ave_intersec = np.array(inter_ave_intersec).mean() if inter_ave_intersec else 0
    print(f'base kn has on average {inter_ave_intersec} inter kn interseciton')

if __name__ == "__main__":
    model_list = [
        "bert-base-cased",
        "bert-large-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "answerdotai/ModernBERT-large",
        "answerdotai/ModernBERT-base",
        # FINETUNED models:
        "aieng-lab/bert-large-cased_requirement-completion",
        "aieng-lab/ModernBERT-large_requirement-completion",
        "aieng-lab/bert-large-cased_incivility",
        "aieng-lab/ModernBERT-large_incivility",
        "aieng-lab/bert-large-cased_tone-bearing",
        "aieng-lab/ModernBERT-large_tone-bearing",
        "aieng-lab/bert-large-cased_sentiment",
        "aieng-lab/ModernBERT-large_sentiment",
        "aieng-lab/bert-large-cased_requirement-type",
        "aieng-lab/ModernBERT-large_requirement-type",
    ]

    for model_name in model_list:
        
        kn_dir = os.path.join("../results", model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name, "kn")
        fig_dir = os.path.join("../results", model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name, "figs")
        if os.path.exists(kn_dir):
            print(f"Analyzing {kn_dir}")
            analyze_kn_dir(kn_dir, fig_dir)
        else:
            print(f"Skipping {kn_dir}, directory does not exist.")