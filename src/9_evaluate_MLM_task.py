import os
import sys
import time
import csv
import pickle
import logging
from typing import List, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer


# ==== Your imports ====
from transformers import BertTokenizerFast
from custom_bert import BertForMaskedLM

sys.path.append("../")
from utils.evaluation import compute_metrics_mlm_hf as compute_metrics

# ==== Config ====
DATASET_PATH = "./datasets/requirement_completion/requirement_completion.pkl"
TASK_NAME = "requirement_completion"
OUTPUT_CSV = "MLM_req-compl_suppression_results.csv"
BATCH_SIZE = 32
MLM_PROB = 0.15
MASK_REPLACE_PROB = 0.8
RAND_REPLACE_PROB = 0.1
TARGET_POS = 16  # VERB

# List your models and BRs here
MODEL_LIST: List[str] = [
        #"bert-large-cased",
        "aieng-lab/bert-large-cased_requirement-completion"
]

BR_LIST: List[str] = ["None", "BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]  # None,BR01..BR09

# Where checkpoints live:
# expects ../results/<model_dirname>/kn_erase-<BR>.pt
# model_dirname is derived from model name by replacing "/" with "_"
CHECKPOINT_ROOT = "../results"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================= Helper functions =======================

def model_to_dirname(model_name: str) -> str:
    """Sanitize a model name/path to a folder name used in results."""
    return model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name

def checkpoint_path_for(model_name: str, br: str) -> str:
    return os.path.join(
        CHECKPOINT_ROOT,
        model_to_dirname(model_name),
        f"kn_erase-{br}.pt"
    )

def mask_tokens_with_pos(inputs, pos_tags, tokenizer, mlm_probability=0.15,
                         mask_replace_prob=0.8, random_replace_prob=0.1,
                         target_pos=16):
    """
    Mask tokens based on POS tags for MLM-style evaluation.
    """
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # Convert list of lists of POS tags to a tensor
    pos_tensor = torch.full_like(input_ids, fill_value=-1)
    for i in range(batch_size):
        pos_tensor[i, :len(pos_tags[i])] = torch.tensor(pos_tags[i], dtype=torch.long, device=pos_tensor.device)

    # Mask only where POS == target_pos
    eligible_mask = (pos_tensor == target_pos)

    # Apply MLM probability to eligible positions
    probability_matrix = torch.full_like(input_ids, fill_value=mlm_probability, dtype=torch.float)
    mask_prob = torch.bernoulli(probability_matrix).bool()
    masked_indices = eligible_mask & mask_prob

    # Create labels: keep original token IDs only at masked positions
    labels[~masked_indices] = -100

    # 80% of masked → [MASK]
    indices_replaced = torch.bernoulli(torch.full_like(input_ids, mask_replace_prob, dtype=torch.float)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of masked → random token
    indices_random = torch.bernoulli(torch.full_like(input_ids, random_replace_prob, dtype=torch.float)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_tokens[indices_random]

    # Remaining 10% → unchanged
    return input_ids, labels

def align_pos_tags_with_tokens(tags, word_ids):
    new_tags = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            tag = -100 if word_id is None else tags[word_id]
            new_tags.append(tag)
        elif word_id is None:
            new_tags.append(-100)
        else:
            tag = tags[word_id]
            new_tags.append(tag)
    return new_tags

def tokenize_and_align_pos_tags(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=False)
    all_pos_tags = examples["pos_tags"]
    new_pos_tags = []
    for i, tags in enumerate(all_pos_tags):
        word_ids = tokenized_inputs.word_ids(i)
        new_pos_tags.append(align_pos_tags_with_tokens(tags, word_ids))
    tokenized_inputs["pos_tags"] = new_pos_tags
    return tokenized_inputs

def group_texts(examples, max_length):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def load_raw_dataset(path: str) -> Dataset:
    logger.info(f"Loading dataset from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    ds = Dataset.from_dict({
        "tokens": [[j[0] for j in i] for i in data],
        "pos_tags": [[j[1] for j in i] for i in data]
    })
    logger.info(f"Raw dataset loaded: {ds}")
    return ds

def prepare_dataset_for_model(raw_ds: Dataset, tokenizer) -> Dataset:
    logger.info("Tokenizing and aligning POS tags...")
    tokenized = raw_ds.map(
        lambda x: tokenize_and_align_pos_tags(x, tokenizer),
        batched=True,
        remove_columns=["tokens", "pos_tags"]
    )
    logger.info("Grouping texts...")
    grouped = tokenized.map(
        lambda x: group_texts(x, tokenizer.model_max_length),
        batched=True
    )
    return grouped

def evaluate_model_on_grouped(grouped: Dataset, model: torch.nn.Module, tokenizer, device: torch.device) -> Dict[str, float]:
    logger.info("Running evaluation...")
    model.eval()
    all_logits = []
    all_labels = []

    for i in range(0, len(grouped), BATCH_SIZE):
        batch = grouped[i:i+BATCH_SIZE]
        input_ids = torch.tensor(batch["input_ids"], device=device)
        attention_mask = torch.tensor(batch["attention_mask"], device=device)
        pos_tags = batch["pos_tags"]  # list of lists

        input_ids_masked, labels = mask_tokens_with_pos(
            {"input_ids": input_ids.clone()},
            pos_tags=pos_tags,
            tokenizer=tokenizer,
            mlm_probability=MLM_PROB,
            mask_replace_prob=MASK_REPLACE_PROB,
            random_replace_prob=RAND_REPLACE_PROB,
            target_pos=TARGET_POS
        )
        with torch.no_grad():
            outputs = model(input_ids=input_ids_masked, attention_mask=attention_mask)
            # custom_bert returns (ffn_weights, tgt_logits)
            logits = outputs[1] if isinstance(outputs, (tuple, list)) else getattr(outputs, "logits", None)
            if logits is not None:
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    if all_logits.size == 0:
        logger.warning("No logits collected; returning empty metrics.")
        return {}

    logger.info("Computing metrics...")
    metrics = compute_metrics((all_logits, all_labels))
    return metrics

def append_csv_row(csv_path: str, fieldnames: List[str], row: Dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ======================= Main pipeline =======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    raw_ds = load_raw_dataset(DATASET_PATH)

    # Prepare CSV header
    # We don't know metric keys beforehand; we’ll detect from first successful run.
    pending_header = True
    fieldnames = None

    for model_name in MODEL_LIST:
        logger.info(f"=== MODEL: {model_name} ===")
        # Tokenizer per model (BERT* uses BertTokenizerFast; AutoTokenizer also works)
        tokenizer = BertTokenizerFast.from_pretrained(model_name) if "bert" in model_name else AutoTokenizer.from_pretrained(model_name)
        grouped = prepare_dataset_for_model(raw_ds, tokenizer)

        for br in BR_LIST:
            if br != "None":
                ckpt_path = checkpoint_path_for(model_name, br)
                if not os.path.exists(ckpt_path):
                    logger.warning(f"[SKIP] Checkpoint not found for {model_name} {br}: {ckpt_path}")
                    continue

            logger.info(f"Loading model + checkpoint for {br if br != 'None' else 'baseline'}")
            model = BertForMaskedLM.from_pretrained(model_name)
            if br != "None":
                state_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
            model.to(device)
            tic = time.perf_counter()
            metrics = evaluate_model_on_grouped(grouped, model, tokenizer, device)
            toc = time.perf_counter()
            logger.info(f"Evaluation time: {toc - tic:.2f} seconds")
            # Free VRAM/CPU mem before next loop
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # If metrics empty, still record a line with NA?
            if not metrics:
                logger.warning(f"No metrics computed for {model_name} {br}; writing NA metrics.")
                row = {
                    "model": model_name,
                    "task": TASK_NAME,
                    "biased_rel_removed": br
                }
                # Create minimal header if needed
                if pending_header:
                    fieldnames = list(row.keys())
                    append_csv_row(OUTPUT_CSV, fieldnames, row)
                    pending_header = False
                else:
                    append_csv_row(OUTPUT_CSV, fieldnames, row)
                continue

            # Build row and header
            row = {
                "model": model_name,
                "task": TASK_NAME,
                "biased_rel_removed": br,
                "time": toc - tic,
                **metrics
            }

            if pending_header:
                fieldnames = ["model", "task", "biased_rel_removed", "time"] + list(metrics.keys())
                append_csv_row(OUTPUT_CSV, fieldnames, row)
                pending_header = False
            else:
                # Ensure stable column order (add any new metric keys if they appear)
                extra_keys = [k for k in metrics.keys() if k not in fieldnames]
                if extra_keys:
                    # Extend header to include new metrics
                    fieldnames += extra_keys
                    # Rewrite CSV with new header (simple approach: read-old, write-new)
                    logger.info(f"Extending CSV header with new metrics: {extra_keys}")
                    with open(OUTPUT_CSV, "r", newline="") as f:
                        reader = list(csv.DictReader(f))
                    with open(OUTPUT_CSV, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for old_row in reader:
                            writer.writerow(old_row)
                append_csv_row(OUTPUT_CSV, fieldnames, row)

            logger.info(f"✅ Done: {model_name} | {br}")

    logger.info("All evaluations completed.")

if __name__ == "__main__":
    main()
