"""
BERT MLM runner
"""

import logging
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

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


def run_modify_activation(
    data_path,
    tmp_data_path,
    bert_model,
    output_dir,
    kn_dir,
    output_prefix='',
    max_seq_length=128,
    do_lower_case=False,
    no_cuda=False,
    gpus='0',
    seed=42,
    debug=-1
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
        device = torch.device("cuda:0")
        n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

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
    if tmp_data_path and os.path.exists(tmp_data_path):
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
            if debug > 0 and len(eval_bag_list_perrel[bag_rel]) >= debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        if tmp_data_path:
            with open(tmp_data_path, 'w') as fw:
                json.dump(eval_bag_list_perrel, fw, indent=2)

    def eval_modification(prefix=''):
        rlt_dict = {}
        for filename in os.listdir(kn_dir):
            if not filename.startswith(f'{prefix}kn_bag-'):
                continue
            relation = filename.split('.')[0].split('-')[-1]
            save_key = filename.split('.')[0]
            print(f'calculating {prefix}relation {relation} ...')
            rlt_dict[save_key] = {
                'own:ori_prob': [],
                'rm_own:ave_delta': [],
                'rm_own:ave_delta_ratio': None,
                'eh_own:ave_delta': [],
                'eh_own:ave_delta_ratio': None,
                'oth:ori_prob': [],
                'rm_oth:ave_delta': [],
                'rm_oth:ave_delta_ratio': None,
                'eh_oth:ave_delta': [],
                'eh_oth:ave_delta_ratio': None,
                'time': None
            }
            with open(os.path.join(kn_dir, filename), 'r') as fr:
                kn_bag_list = json.load(fr)
            # record running time
            tic = time.perf_counter()
            for bag_idx, kn_bag in enumerate(kn_bag_list):
                if (bag_idx + 1) % 100 == 0:
                    print(f'calculating {prefix}bag {bag_idx} ...')
                # =============== eval own bag: remove & enhance ================
                eval_bag = eval_bag_list_perrel[relation][bag_idx]
                for eval_example in eval_bag:
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

                    # record [MASK]'s gold label
                    gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])

                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                    ori_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar

                    # remove
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='remove')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['own:ori_prob'].append(ori_gold_prob.item())
                    rlt_dict[save_key]['rm_own:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                    # enhance
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='enhance')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['eh_own:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                # =============== eval another bag: remove & enhance ================
                oth_relations = list(eval_bag_list_perrel.keys())
                oth_relations = [rel for rel in oth_relations if rel != relation]
                oth_relation = random.choice(oth_relations)
                oth_idx = random.randint(0, len(eval_bag_list_perrel[oth_relation]) - 1)
                eval_bag = eval_bag_list_perrel[oth_relation][oth_idx]
                for eval_example in eval_bag:
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

                    # record [MASK]'s gold label
                    gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])

                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                    ori_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar

                    # remove
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='remove')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['oth:ori_prob'].append(ori_gold_prob.item())
                    rlt_dict[save_key]['rm_oth:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

                    # enhance
                    _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='enhance')  # (1, n_vocab)
                    int_gold_prob = F.softmax(logits, dim=-1)[0, gold_label]  # scalar
                    rlt_dict[save_key]['eh_oth:ave_delta'].append((int_gold_prob - ori_gold_prob).item())

            # record running time
            toc = time.perf_counter()
            logger.info(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")
            
            for k, v in rlt_dict[save_key].items():
                if rlt_dict[save_key][k] is not None and len(rlt_dict[save_key][k]) > 0:
                    rlt_dict[save_key][k] = np.array(rlt_dict[save_key][k]).mean()
            rlt_dict[save_key]['rm_own:ave_delta_ratio'] = rlt_dict[save_key]['rm_own:ave_delta'] / rlt_dict[save_key]['own:ori_prob']
            rlt_dict[save_key]['eh_own:ave_delta_ratio'] = rlt_dict[save_key]['eh_own:ave_delta'] / rlt_dict[save_key]['own:ori_prob']
            rlt_dict[save_key]['rm_oth:ave_delta_ratio'] = rlt_dict[save_key]['rm_oth:ave_delta'] / rlt_dict[save_key]['oth:ori_prob']
            rlt_dict[save_key]['eh_oth:ave_delta_ratio'] = rlt_dict[save_key]['eh_oth:ave_delta'] / rlt_dict[save_key]['oth:ori_prob']
            rlt_dict[save_key]['time'] = toc - tic
            print(save_key, '==============>', rlt_dict[save_key])

        with open(os.path.join(kn_dir, f'{prefix}modify_activation_rlt.json'), 'w') as fw:
            json.dump(rlt_dict, fw, indent=2)

    eval_modification('')
    eval_modification('base_')


if __name__ == "__main__":
    # Example usage for multiple models
    model_list = [
        #"bert-base-cased",
        #"bert-large-cased",
        #"bert-base-uncased",
        #"bert-large-uncased",
        #"answerdotai/ModernBERT-large",
        #"answerdotai/ModernBERT-base",
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
        output_prefix = ""
        max_seq_length = 128
        do_lower_case = True if 'uncased' in model_name else False
        no_cuda = False
        gpus = '1'
        seed = 42
        debug = 100000

        if not os.path.exists(kn_dir):
            print(f"Skipping {model_name}: kn_dir not found.")
            continue

        run_modify_activation(
            data_path=data_path,
            tmp_data_path=tmp_data_path,
            bert_model=bert_model,
            output_dir=output_dir,
            kn_dir=kn_dir,
            output_prefix=output_prefix,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            no_cuda=no_cuda,
            gpus=gpus,
            seed=seed,
            debug=debug
        )