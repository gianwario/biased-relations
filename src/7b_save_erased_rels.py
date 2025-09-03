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
         pt_relation=None
         ):

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




    def erase(rel):
        print(f'evaluating {rel}...')
        with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'r') as fr:
            kn_bag_list = json.load(fr)
        tic = time.perf_counter()
        # ======================== calculate kn_rel =================================
        kn_rel = []
        kn_counter = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_counter.update([pos_list2str(kn)])
        most_common_kn = kn_counter.most_common(20)
        print(most_common_kn)
        kn_rel = [pos_str2list(kn_str[0]) for kn_str in most_common_kn]

        # ======================== load model =================================
        model = BertForMaskedLM.from_pretrained(bert_model)
        model.to(device)
        model.eval()

        # ========================== eval self =====================
        correct = 0
        total = 0
        log_ppl_list = []
        for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[rel]):
            print(f'evaluating ori {bag_idx} / {len(eval_bag_list_perrel[rel])}')
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
                    return
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                log_ppl = np.log(1.0 / gold_prob.item())
                log_ppl_list.append(log_ppl)
                ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                total += 1
                if ori_pred_label == eval_example[1]:
                    correct += 1
        ppl = np.exp(np.array(log_ppl_list).mean())
        acc = correct / total
        # ========================== eval other =====================
        o_correct = 0
        o_total = 0
        o_log_ppl_list = []
        for o_rel, eval_bag_list in eval_bag_list_perrel.items():
            if o_rel == rel:
                continue
            print(f'evaluating for another relation {o_rel}')
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                # if bag_idx % 100 != 0:
                #     continue
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
                        return
                    gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                    ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                    gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                    o_log_ppl = np.log(1.0 / gold_prob.item())
                    o_log_ppl_list.append(o_log_ppl)
                    ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                    o_total += 1
                    if ori_pred_label == eval_example[1]:
                        o_correct += 1
        o_ppl = np.exp(np.array(o_log_ppl_list).mean())
        o_acc = o_correct / o_total

        # ============================================== erase knowledge begin ===========================================================
        print(f'-- kn_num: {len(kn_rel)}')
        # unk_emb = model.bert.embeddings.word_embeddings.weight[100]

        with torch.no_grad():

            for layer, pos in kn_rel:
                # model.bert.encoder.layer[layer].output.dense.weight[:, pos] = unk_emb
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] = 0
            # save model
        model_save_path = os.path.join(output_dir, f'kn_erase-{rel}.pt')
        torch.save(model.state_dict(), model_save_path)
        print(f'-- model saved to {model_save_path}')
        # ============================================== erase knowledge end =============================================================
        
        # ========================== eval self =====================
        new_correct = 0
        new_total = 0
        new_log_ppl_list = []
        for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[rel]):
            print(f'evaluating erased {bag_idx} / {len(eval_bag_list_perrel[rel])}')
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
                    return
                gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                new_log_ppl = np.log(1.0 / gold_prob.item())
                new_log_ppl_list.append(new_log_ppl)
                ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                new_total += 1
                if ori_pred_label == eval_example[1]:
                    new_correct += 1
        new_ppl = np.exp(np.array(new_log_ppl_list).mean())
        new_acc = new_correct / new_total

        # ========================== eval other =====================
        o_new_correct = 0
        o_new_total = 0
        o_new_log_ppl_list = []
        for o_rel, eval_bag_list in eval_bag_list_perrel.items():
            if o_rel == rel:
                continue
            print(f'evaluating for another relation {o_rel}')
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                # if bag_idx % 100 != 0:
                #     continue
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
                        return
                    gold_id = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                    ori_pred_prob, ori_pred_label_id = F.softmax(logits, dim=-1)[0].max(dim=-1)
                    gold_prob = F.softmax(logits, dim=-1)[0][gold_id]
                    o_new_log_ppl = np.log(1.0 / gold_prob.item())
                    o_new_log_ppl_list.append(o_new_log_ppl)
                    ori_pred_label = tokenizer.convert_ids_to_tokens(ori_pred_label_id.item())
                    o_new_total += 1
                    if ori_pred_label == eval_example[1]:
                        o_new_correct += 1
        o_new_ppl = np.exp(np.array(o_new_log_ppl_list).mean())
        o_new_acc = o_new_correct / o_new_total
        toc = time.perf_counter()
        print(f'======================================== {rel} ===========================================')
        print(f'original accuracy: {acc:.4}')
        print(f'erased accuracy: {new_acc:.4}')
        if acc == 0.0:
            erased_ratio = 0.0
        else:
            erased_ratio = (acc - new_acc) / acc

        print(f'erased ratio: {erased_ratio:.4}')
        print(f'# Kneurons: {len(kn_rel)}')

        print(f'original ppl: {ppl:.4}')
        print(f'erased ppl: {new_ppl:.4}')
        if ppl == 0.0:
            erased_ratio = 0.0
        else:
            erased_ratio = (ppl - new_ppl) / ppl
        print(f'ppl increasing ratio: {erased_ratio:.4}')

        print(f'(for other) original accuracy: {o_acc:.4}')
        print(f'(for other) erased accuracy: {o_new_acc:.4}')
        if o_acc == 0.0:
            o_erased_ratio = 0.0
        else:
            o_erased_ratio = (o_acc - o_new_acc) / o_acc
        print(f'(for other) erased ratio: {o_erased_ratio:.4}')

        print(f'(for other) original ppl: {o_ppl:.4}')
        print(f'(for other) erased ppl: {o_new_ppl:.4}')
        if o_ppl == 0.0: 
            o_erased_ratio = 0.0
        else:
            o_erased_ratio = (o_ppl - o_new_ppl) / o_ppl
        print(f'(for other) ppl increasing ratio: {o_erased_ratio:.4}')
        
        # Inserted block to save per-relation evaluation results to JSON
        result_json = {
            "relation": rel,
            "original_accuracy": round(acc, 4),
            "erased_accuracy": round(new_acc, 4),
            "erased_ratio_accuracy": 0.0 if acc == 0.0 else round((acc - new_acc) / acc, 4),
            "num_kn_rel": len(kn_rel),
            "original_ppl": round(ppl, 4),
            "erased_ppl": round(new_ppl, 4),
            "ppl_increase_ratio": 0.0 if ppl == 0.0 else round((new_ppl - ppl) / ppl, 4),
            "other_original_accuracy": round(o_acc, 4),
            "other_erased_accuracy": round(o_new_acc, 4),
            "other_erased_ratio_accuracy": 0.0 if o_acc == 0.0 else round((o_acc - o_new_acc) / o_acc, 4),
            "other_original_ppl": round(o_ppl, 4),
            "other_erased_ppl": round(o_new_ppl, 4),
            "other_ppl_increase_ratio": 0.0 if o_ppl == 0.0 else round((o_new_ppl - o_ppl) / o_ppl, 4),
            "time":toc-tic
        }

        result_path = os.path.join(output_dir, f"erasing_summary_{rel}.json")
        with open(result_path, 'w') as f:
            json.dump(result_json, f, indent=2)    
    # ======================== erase relations =================================
    if pt_relation is not None:
        erase(pt_relation)
        
        
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
    
    brs = ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]

    for model_name in model_list:
        for br in brs:
            print(f"Running for model: {model_name}, relation: {br}")
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
                debug=debug,
                pt_relation=br
            )
        