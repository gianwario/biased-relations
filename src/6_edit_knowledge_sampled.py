"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
from pprint import pprint
from collections import Counter

import transformers
from transformers import AutoTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features using proper tokenization and masking support"""
    
    # Replace [MASK] placeholder with the tokenizer's actual mask token
    text = example[0].replace("[MASK]", tokenizer.mask_token)

    # Tokenize the input properly with encoding
    enc = tokenizer.encode_plus(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",  # returns tensors, we convert them to list below
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    # Extract input features
    input_ids = enc["input_ids"][0].tolist()
    input_mask = enc["attention_mask"][0].tolist()
    segment_ids = enc["token_type_ids"][0].tolist()
    
    # Baseline uses [UNK] everywhere (same length as input)
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    baseline_ids = [unk_id] * len(input_ids)

    # Token-level info (for display/logging, not for indexing)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Construct the features dict
    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }

    # Token info (optional, for logging or display purposes)
    tokens_info = {
        "tokens": tokens,
        "relation": example[2],
        "gold_obj": example[1],
        "pred_obj": None
    }

    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def main(bert_model,
         data_path,
         tmp_data_path,
         kn_dir,
         output_dir,
         gpus,
         max_seq_length,
         debug,
         do_lower_case,
         no_cuda=False,
         seed=42,
         norm_lambda1=1,
         norm_lambda2=8):

    # set device
    if no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(gpus) == 1:
        device = torch.device("cuda:%s" % gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # save args
    os.makedirs(output_dir, exist_ok=True)

    # init tokenizer
    tokenizer_name = bert_model
    if "ModernBERT-large" in model_name:
        tokenizer_name = "answerdotai/ModernBERT-large"
    if "ModernBERT-base" in model_name:
        tokenizer_name = "answerdotai/ModernBERT-base"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case, force_download=True)


    # Load pre-trained BERT
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(bert_model)
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    if os.path.exists(tmp_data_path):
        with open(tmp_data_path, 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else:
        with open(data_path, 'r') as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(tmp_data_path, 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)


    def select_kn(kn_bag, kn_bag_list):
        kn_counter = Counter()
        for kn in kn_bag:
            kn_counter.update([pos_list2str(kn)])
        for tmp_kn_bag in kn_bag_list:
            for kn in tmp_kn_bag:
                str_kn = pos_list2str(kn)
                if str_kn not in kn_counter:
                    continue
                kn_counter.update([pos_list2str(kn)])
        new_kn_bag = []
        for k, v in kn_counter.items():
            if v > len(kn_bag_list) * 0.1:
                continue
            new_kn_bag.append(pos_str2list(k))
        # random kn
        # for i in range(len(new_kn_bag)):
        #     new_kn_bag[i] = [random.randint(0, 11), random.randint(0, 3071)]
        return new_kn_bag


    def edit(rel, bag_idx, tgt_ent):
        eval_bag = eval_bag_list_perrel[rel][bag_idx]
    
        with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'r') as fr:
            kn_bag_list = json.load(fr)
        

        if bag_idx >= len(kn_bag_list):
            print(f"[WARN] Skipping bag_idx {bag_idx} for relation {rel} — not in kn_bag_list.")
            return None

        kn_bag = kn_bag_list[bag_idx]
        kn_bag = select_kn(kn_bag, kn_bag_list)
        print(kn_bag)

        results = {
            'success_updated': 0,
            'success_updated_5': 0,
            'ori_changed': 0,
            'tgt_prob_inc': 0,
            'tgt_prob_inc_ratio': 0,
            'tgt_ori_rank': 0,
            'tgt_new_rank': 0,
            'tgt_rank_inc': 0,
            'ori_inter_log_ppl': [],
            'ori_inner_log_ppl': [],
            'new_inter_log_ppl': [],
            'new_inner_log_ppl': [],
            'time':None,
            'rel': rel,
        }

        if len(kn_bag) == 0:
            return 'no_kn'

        rd_idx = random.randint(1, len(eval_bag)) - 1

        tic = time.perf_counter()
        eval_example = eval_bag[rd_idx]
        eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
        # convert features to long type tensors
        baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
        baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        baseline_ids = baseline_ids.to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        # record [MASK]'s position
        mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        try:
            tgt_pos = eval_features['input_ids'].index(mask_token_id)
        except ValueError:
            print("Warning: [MASK] token ID not found in input_ids:", eval_features['input_ids'])
            return
    
        # record [MASK]'s gold label
        tgt_label_id = tokenizer.convert_tokens_to_ids(tgt_ent)

        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
        ori_tgt_prob = F.softmax(logits, dim=-1)[0, tgt_label_id]  # scalar
        ori_tgt_rank = logits.squeeze().argsort(descending=True).argsort()[tgt_label_id] + 1
        ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
        ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())

        if ori_pred_label != eval_example[1]:
            return 'incorrect_answer'

        # ori pred label, ori pred prob, tgt label, ori tgt prob
        print(f'{rel}-{bag_idx}, # Kneurons: {len(kn_bag)},', 'example:', eval_example[0], 'gold:', eval_example[1])
        print('============================== ori =========================================')
        print(f'ori pred label: {ori_pred_label}, ori pred label prob: {ori_pred_prob:.8}')
        print(f'tgt label: {tgt_ent}, tgt prob: {ori_tgt_prob:.8}')

        # edit knowledge
        ori_pred_emb = model.bert.embeddings.word_embeddings.weight[ori_pred_label_id.item()]
        tgt_emb = model.bert.embeddings.word_embeddings.weight[tgt_label_id]
        print(f'-- kn_num: {len(kn_bag)}')
        lambda_list_1 = []
        lambda_list_2 = []
        ori_pred_emb_norm = torch.norm(ori_pred_emb)
        tgt_emb_norm = torch.norm(tgt_emb)
        for layer, pos in kn_bag:
            value_norm = torch.norm(model.bert.encoder.layer[layer].output.dense.weight[:, pos])
            lambda_list_1.append(value_norm / ori_pred_emb_norm * norm_lambda1)
            lambda_list_2.append(value_norm / tgt_emb_norm * norm_lambda2)
        with torch.no_grad():
            for i, (layer, pos) in enumerate(kn_bag):
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= ori_pred_emb * lambda_list_1[i]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] += tgt_emb * lambda_list_2[i]

        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
        new_tgt_prob = F.softmax(logits, dim=-1)[0, tgt_label_id]  # scalar
        new_tgt_rank = logits.squeeze().argsort(descending=True).argsort()[tgt_label_id] + 1
        new_ori_pred_prob = F.softmax(logits, dim=-1)[0, ori_pred_label_id]  # scalar
        new_pred_prob, new_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
        new_pred_label = tokenizer.convert_ids_to_tokens(new_pred_label_id.item())

        # new pred label, new pred prob, tgt label, new tgt prob
        print('============================== edited =========================================')
        print(f'ori pred label: {ori_pred_label}, new ori pred label prob: {new_ori_pred_prob:.8}')
        print(f'new pred label: {new_pred_label}, new pred label prob: {new_pred_prob:.8}')
        print(f'tgt label: {tgt_ent}, tgt prob: {new_tgt_prob:.8}')

        # recover knowledge
        with torch.no_grad():
            for i, (layer, pos) in enumerate(kn_bag):
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= ori_pred_emb * lambda_list_1[i]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] += tgt_emb * lambda_list_2[i]

        if new_pred_label == tgt_ent:
            results['success_updated'] = 1
        if new_tgt_rank.item() <= 5:
            results['success_updated_5'] = 1
        if new_pred_label != ori_pred_label:
            results['ori_changed'] = 1
        results['tgt_prob_inc'] = new_tgt_prob.item() - ori_tgt_prob.item()
        results['tgt_prob_inc_ratio'] = (new_tgt_prob.item() - ori_tgt_prob.item()) / ori_tgt_prob.item()
        results['tgt_ori_rank'] = ori_tgt_rank.item()
        results['tgt_new_rank'] = new_tgt_rank.item()
        results['tgt_rank_inc'] = ori_tgt_rank.item() - new_tgt_rank.item()

        # =========================================================================== calculate PPL ===========================================================================
        # inner relation
        inner_log_ppl_list = []
        eval_bag_list = eval_bag_list_perrel[rel]
        inner_rand_bag_idx_list = []
        for i in range(5):
            rand_bag_idx = random.randint(0, len(eval_bag_list) - 1)
            while rand_bag_idx == bag_idx:
                rand_bag_idx = random.randint(0, len(eval_bag_list) - 1)
            inner_rand_bag_idx_list.append(rand_bag_idx)
        for i in range(5):
            rand_bag_idx = inner_rand_bag_idx_list[i]
            eval_bag = eval_bag_list[rand_bag_idx]
            for idx, eval_example in enumerate(eval_bag):
                eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # record [MASK]'s position
                mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                try:
                    tgt_pos = eval_features['input_ids'].index(mask_token_id)
                except ValueError:
                    print("Warning: [MASK] token ID not found in input_ids:", eval_features['input_ids'])
                    continue
    
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                inner_log_ppl = np.log(1.0 / gold_prob.item())
                inner_log_ppl_list.append(inner_log_ppl)
        results['ori_inner_log_ppl'] = np.mean(inner_log_ppl_list)
        # inter relation
        inter_log_ppl_list = []
        rels = list(eval_bag_list_perrel.keys())
        inter_rand_rel_list = []
        inter_rand_bag_idx_list = []
        for i in range(5):
            rand_rel = random.choice(rels)
            while rand_rel == rel:
                rand_rel = random.choice(rels)
            rand_bag_idx = random.randint(0, len(eval_bag_list_perrel[rand_rel]) - 1)
            inter_rand_rel_list.append(rand_rel)
            inter_rand_bag_idx_list.append(rand_bag_idx)
        for i in range(5):
            rand_rel = inter_rand_rel_list[i]
            eval_bag_list = eval_bag_list_perrel[rand_rel]
            rand_bag_idx = inter_rand_bag_idx_list[i]
            eval_bag = eval_bag_list[rand_bag_idx]
            for idx, eval_example in enumerate(eval_bag):
                eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # record [MASK]'s position
                mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                try:
                    tgt_pos = eval_features['input_ids'].index(mask_token_id)
                except ValueError:
                    print("Warning: [MASK] token ID not found in input_ids:", eval_features['input_ids'])
                    continue
    
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                inter_log_ppl = np.log(1.0 / gold_prob.item())
                inter_log_ppl_list.append(inter_log_ppl)
        results['ori_inter_log_ppl'] = np.mean(inter_log_ppl_list)

        # edit knowledge
        ori_pred_emb = model.bert.embeddings.word_embeddings.weight[ori_pred_label_id.item()]
        tgt_emb = model.bert.embeddings.word_embeddings.weight[tgt_label_id]
        print(f'-- kn_num: {len(kn_bag)}')
        with torch.no_grad():

            for i, (layer, pos) in enumerate(kn_bag):
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= ori_pred_emb * lambda_list_1[i]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] += tgt_emb * lambda_list_2[i]

        # inner relation
        inner_log_ppl_list = []
        eval_bag_list = eval_bag_list_perrel[rel]
        for i in range(5):
            rand_bag_idx = inner_rand_bag_idx_list[i]
            eval_bag = eval_bag_list[rand_bag_idx]
            for idx, eval_example in enumerate(eval_bag):
                eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # record [MASK]'s position
                if '[MASK]' not in tokens_info['tokens']:
                    print("Warning: [MASK] not found in tokens:", tokens_info['tokens'])
                    continue
                tgt_pos = tokens_info['tokens'].index('[MASK]')
    
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                inner_log_ppl = np.log(1.0 / gold_prob.item())
                inner_log_ppl_list.append(inner_log_ppl)
        results['new_inner_log_ppl'] = np.mean(inner_log_ppl_list)
        # inter relation
        inter_log_ppl_list = []
        rels = list(eval_bag_list_perrel.keys())
        for i in range(5):
            rand_rel = inter_rand_rel_list[i]
            eval_bag_list = eval_bag_list_perrel[rand_rel]
            rand_bag_idx = inter_rand_bag_idx_list[i]
            eval_bag = eval_bag_list[rand_bag_idx]
            for idx, eval_example in enumerate(eval_bag):
                eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # record [MASK]'s position
                if '[MASK]' not in tokens_info['tokens']:
                    print("Warning: [MASK] not found in tokens:", tokens_info['tokens'])
                    continue
                tgt_pos = tokens_info['tokens'].index('[MASK]')
    
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                inter_log_ppl = np.log(1.0 / gold_prob.item())
                inter_log_ppl_list.append(inter_log_ppl)
        results['new_inter_log_ppl'] = np.mean(inter_log_ppl_list)

        # recover knowledge
        with torch.no_grad():

            for i, (layer, pos) in enumerate(kn_bag):
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] += ori_pred_emb * lambda_list_1[i]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= tgt_emb * lambda_list_2[i]
        toc = time.perf_counter()
        results['time'] = toc - tic
        return results

    rels = list(eval_bag_list_perrel.keys())
    ave_results = {
        'success_updated': [],
        'success_updated_5': [],
        'ori_changed': [],
        'tgt_prob_inc': [],
        'tgt_prob_inc_ratio': [],
        'tgt_ori_rank': [],
        'tgt_new_rank': [],
        'tgt_rank_inc': [],
        'other_ppl_inc': [],
        'ori_inner_log_ppl': [],
        'new_inner_log_ppl': [],
        'ori_inter_log_ppl': [],
        'new_inter_log_ppl': []
    }
    success_updated = []
    ori_changed = []
    tgt_prob_inc = []
    tgt_prob_inc_ratio = []
    tgt_ori_rank = []
    tgt_new_rank = []
    tgt_rank_inc = []
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' * 2)
    for rel in rels:
        rel_sample_cnt = 0
        for i in range(10000):
            sampled_bag_idx = random.randint(0, len(eval_bag_list_perrel[rel]) - 1)
            while True:
                replaced_tail = random.choice(eval_bag_list_perrel[rel])[0][1]
                if replaced_tail != eval_bag_list_perrel[rel][sampled_bag_idx][0][1]:
                    break
            results = edit(rel, sampled_bag_idx, replaced_tail)
            with open(os.path.join(kn_dir, f'{rel}_edit.json'), 'w') as fw:
                json.dump(results, fw, indent=2)
            if results is None or isinstance(results, str):
                continue
            rel_sample_cnt += 1
            ave_results['success_updated'].append(results['success_updated'])
            ave_results['success_updated_5'].append(results['success_updated_5'])
            ave_results['ori_changed'].append(results['ori_changed'])
            ave_results['tgt_prob_inc'].append(results['tgt_prob_inc'])
            ave_results['tgt_prob_inc_ratio'].append(results['tgt_prob_inc_ratio'])
            ave_results['tgt_ori_rank'].append(results['tgt_ori_rank'])
            ave_results['tgt_new_rank'].append(results['tgt_new_rank'])
            ave_results['tgt_rank_inc'].append(results['tgt_rank_inc'])
            ave_results['ori_inner_log_ppl'].append(results['ori_inner_log_ppl'])
            ave_results['new_inner_log_ppl'].append(results['new_inner_log_ppl'])
            ave_results['ori_inter_log_ppl'].append(results['ori_inter_log_ppl'])
            ave_results['new_inter_log_ppl'].append(results['new_inter_log_ppl'])
            pprint(results)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' * 2)
            if rel_sample_cnt == 10:
                break
        print(f'@@@@@ {rel} has {rel_sample_cnt} samples.')
    ori_MRR = (1 / np.array(ave_results['tgt_ori_rank'])).mean()
    new_MRR = (1 / np.array(ave_results['tgt_new_rank'])).mean()
    ave_results['success_updated'] = np.array(ave_results['success_updated']).mean()
    ave_results['success_updated_5'] = np.array(ave_results['success_updated_5']).mean()
    ave_results['ori_changed'] = np.array(ave_results['ori_changed']).mean()
    ave_results['tgt_prob_inc'] = np.array(ave_results['tgt_prob_inc']).mean()
    ave_results['tgt_prob_inc_ratio'] = np.array(ave_results['tgt_prob_inc_ratio']).mean()
    ave_results['tgt_ori_rank'] = np.array(ave_results['tgt_ori_rank']).mean()
    ave_results['tgt_new_rank'] = np.array(ave_results['tgt_new_rank']).mean()
    ave_results['tgt_rank_inc'] = np.array(ave_results['tgt_rank_inc']).mean()
    ave_results['ori_inner_log_ppl'] = np.array(ave_results['ori_inner_log_ppl']).mean()
    ave_results['new_inner_log_ppl'] = np.array(ave_results['new_inner_log_ppl']).mean()
    ave_results['ori_inter_log_ppl'] = np.array(ave_results['ori_inter_log_ppl']).mean()
    ave_results['new_inter_log_ppl'] = np.array(ave_results['new_inter_log_ppl']).mean()
    ave_results['ori_inner_ppl'] = np.exp(ave_results['ori_inner_log_ppl'])
    ave_results['new_inner_ppl'] = np.exp(ave_results['new_inner_log_ppl'])
    ave_results['ori_inter_ppl'] = np.exp(ave_results['ori_inter_log_ppl'])
    ave_results['new_inter_ppl'] = np.exp(ave_results['new_inter_log_ppl'])
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' * 2)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' * 2)
    pprint(ave_results)
    print(f'MR: {ave_results["tgt_ori_rank"]:.8} -> {ave_results["tgt_new_rank"]:.8} (up {ave_results["tgt_new_rank"] - ave_results["tgt_ori_rank"]:.8})')
    print(f'MRR: {ori_MRR:.8} -> {new_MRR:.8} (up {new_MRR - ori_MRR:.8})')
    print(f"inner PPL: {ave_results['ori_inner_ppl']:.8} -> {ave_results['new_inner_ppl']:.8} (up {ave_results['new_inner_ppl'] - ave_results['ori_inner_ppl']:.8})")
    print(f"inter PPL: {ave_results['ori_inter_ppl']:.8} -> {ave_results['new_inter_ppl']:.8} (up {ave_results['new_inter_ppl'] - ave_results['ori_inter_ppl']:.8})")
    with open(os.path.join(kn_dir, f'average_edit.json'), 'w') as fw:
        json.dump(ave_results, fw, indent=2)

if __name__ == "__main__":
    model_list = [
        #"bert-base-cased",
        #"bert-large-cased",
        #"bert-base-uncased",
        #"bert-large-uncased",
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
        print(f"Running for model: {model_name}")
        data_path = ""
        tmp_data_path = f"../data/biased_relations/biased_relations_all_bags.json"
        bert_model = model_name
        output_dir = f"../results/{model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name}"
        kn_dir = f"../results/{model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name}/kn"
        max_seq_length = 128
        do_lower_case = True if 'uncased' in model_name else False
        no_cuda = False
        gpus = '1'
        seed = 42
        debug = 100000

        if not os.path.exists(kn_dir):
            print(f"Skipping {model_name}: kn_dir not found.")
            continue

        main(
            data_path=data_path,
            tmp_data_path=tmp_data_path,
            bert_model=bert_model,
            output_dir=output_dir,
            kn_dir=kn_dir,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            no_cuda=no_cuda,
            gpus=gpus,
            seed=seed,
            debug=debug
        )