import jsonlines, json
import numpy as np
from collections import Counter
import os

import matplotlib.pyplot as plt
import seaborn as sns

threshold_ratio = 0.2
mode_ratio_bag = 0.7
mode_ratio_rel = 0.1

def re_filter(metric_triplets):
    metric_max = -999
    for i in range(len(metric_triplets)):
        metric_max = max(metric_max, metric_triplets[i][2])
    metric_triplets = [triplet for triplet in metric_triplets if triplet[2] >= metric_max * threshold_ratio]
    return metric_triplets

def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])

def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]

def parse_kn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
    mode_threshold = tot_num * mode_ratio
    mode_threshold = max(mode_threshold, min_threshold)
    kn_bag = []
    for pos_str, cnt in pos_cnt.items():
        if cnt >= mode_threshold:
            kn_bag.append(pos_str2list(pos_str))
    return kn_bag

def analysis_file(filename, rlts_dir, metric='ig_gold'):
    rel = filename.split('.')[0].split('-')[-1]
    print(f'===========> parsing important position in {rel}..., mode_ratio_bag={mode_ratio_bag}')

    rlts_bag_list = []
    with open(os.path.join(rlts_dir, filename), 'r', encoding='utf-8') as fr:
        for rlts_bag in jsonlines.Reader(fr):
            rlts_bag_list.append(rlts_bag)

    ave_kn_num = 0
    kn_bag_list = []
    # get imp pos by bag_ig
    for bag_idx, rlts_bag in enumerate(rlts_bag_list):
        pos_cnt_bag = Counter()
        for rlt in rlts_bag:
            res_dict = rlt[1]
            metric_triplets = re_filter(res_dict[metric])
            for metric_triplet in metric_triplets:
                pos_cnt_bag.update([pos_list2str(metric_triplet[:2])])
        kn_bag = parse_kn(pos_cnt_bag, len(rlts_bag), mode_ratio_bag, 3)
        ave_kn_num += len(kn_bag)
        kn_bag_list.append(kn_bag)

    ave_kn_num /= len(rlts_bag_list)

    # get imp pos by rel_ig
    pos_cnt_rel = Counter()
    for kn_bag in kn_bag_list:
        for kn in kn_bag:
            pos_cnt_rel.update([pos_list2str(kn)])
    kn_rel = parse_kn(pos_cnt_rel, len(kn_bag_list), mode_ratio_rel)

    return ave_kn_num, kn_bag_list, kn_rel

def stat(data, pos_type, rel):
    if pos_type == 'kn_rel':
        print(f'{rel}\'s {pos_type} has {len(data)} imp pos. ')
        return
    ave_len = 0
    for kn_bag in data:
        ave_len += len(kn_bag)
    ave_len /= len(data)
    print(f'{rel}\'s {pos_type} has on average {ave_len} imp pos. ')

def process_results_dir(results_root):
    for root, dirs, files in os.walk(results_root):
        kn_dir = os.path.join(root, 'kn')
        if not os.path.exists(kn_dir):
            os.makedirs(kn_dir)
        
        print(f'kn_dir: {kn_dir}')

        for filename in files:
            if filename.endswith('.rlt.jsonl'):
                global threshold_ratio, mode_ratio_bag
                threshold_ratio = 0.2
                mode_ratio_bag = 0.7
                for max_it in range(6):
                    ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename, root)
                    if ave_kn_num < 2:
                        mode_ratio_bag -= 0.05
                    if ave_kn_num > 5:
                        mode_ratio_bag += 0.05
                    if ave_kn_num >= 2 and ave_kn_num <= 5:
                        break
                rel = filename.split('.')[0].split('-')[-1]
                stat(kn_bag_list, 'kn_bag', rel)
                stat(kn_rel, 'kn_rel', rel)
                with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'w') as fw:
                    json.dump(kn_bag_list, fw, indent=2)
                with open(os.path.join(kn_dir, f'kn_rel-{rel}.json'), 'w') as fw:
                    json.dump(kn_rel, fw, indent=2)

                threshold_ratio = 0.5
                mode_ratio_bag = 0.7
                for max_it in range(6):
                    ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename, root, 'base')
                    if ave_kn_num < 2:
                        mode_ratio_bag -= 0.05
                    if ave_kn_num > 5:
                        mode_ratio_bag += 0.05
                    if ave_kn_num >= 2 and ave_kn_num <= 5:
                        break
                rel = filename.split('.')[0].split('-')[-1]
                stat(kn_bag_list, 'kn_bag', rel)
                stat(kn_rel, 'kn_rel', rel)
                with open(os.path.join(kn_dir, f'base_kn_bag-{rel}.json'), 'w') as fw:
                    json.dump(kn_bag_list, fw, indent=2)
                with open(os.path.join(kn_dir, f'base_kn_rel-{rel}.json'), 'w') as fw:
                    json.dump(kn_rel, fw, indent=2)

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
        results_root = f"../results/{model_name}"
        print(f"Processing results for model: {model_name}")
        process_results_dir(results_root)