import pickle
import sys
import logging
import numpy as np
from transformers import BertTokenizerFast
from custom_bert import BertForMaskedLM

import csv
from datasets import Dataset
import torch

sys.path.append("../")
from utils.evaluation import compute_metrics_mlm_hf as compute_metrics

DATASET_PATH = "./datasets/requirement_completion/requirement_completion.pkl"
MODEL_NAME_OR_PATH = "bert-large-cased"  # Change to your model or local path
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def mask_tokens_with_pos(inputs, pos_tags, tokenizer, mlm_probability=0.15, mask_replace_prob=0.8, random_replace_prob=0.1, target_pos=16):
    """
    Mask tokens based on POS tags for MLM-style evaluation.

    Args:
        inputs: dict with input_ids and attention_mask (torch.Tensor)
        pos_tags: list of lists of POS tags (same shape as input_ids)
        tokenizer: the tokenizer used (for special tokens and mask token)
        mlm_probability: how many eligible tokens to mask (e.g., 0.15)
        mask_replace_prob: how many of the masked tokens to replace with [MASK]
        random_replace_prob: how many to replace with a random token
        target_pos: POS tag to consider for masking (default = 16 = VERB)
    Returns:
        inputs["input_ids"] (with masking applied), labels (with -100 in ignored positions)
    """

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # Convert list of lists of POS tags to a tensor
    pos_tensor = torch.full_like(input_ids, fill_value=-1)
    for i in range(batch_size):
        pos_tensor[i, :len(pos_tags[i])] = torch.tensor(pos_tags[i])

    # Mask only where POS == target_pos
    eligible_mask = (pos_tensor == target_pos)

    # Apply MLM probability to eligible positions
    probability_matrix = torch.full_like(input_ids, fill_value=mlm_probability, dtype=torch.float)
    mask_prob = torch.bernoulli(probability_matrix).bool()
    masked_indices = eligible_mask & mask_prob

    # Create labels: keep original token IDs only at masked positions
    labels[~masked_indices] = -100

    # Apply BERT-style masking:
    # 80% of masked → [MASK]
    indices_replaced = torch.bernoulli(torch.full_like(input_ids, mask_replace_prob, dtype=torch.float)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of masked → random token
    indices_random = torch.bernoulli(torch.full_like(input_ids, random_replace_prob, dtype=torch.float)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # Remaining 10% → unchanged (input_ids stay as is)

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
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)
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

def main():
    logger.info(f"Trying to load dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    # Use the whole dataset
    logger.info("Preparing dataset...")
    dataset = Dataset.from_dict({
        "tokens": [[j[0] for j in i] for i in data],
        "pos_tags": [[j[1] for j in i] for i in data]
    })
    print(dataset)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME_OR_PATH)

    logger.info("Tokenizing and aligning POS tags...")
    tokenized = dataset.map(
        lambda x: tokenize_and_align_pos_tags(x, tokenizer),
        batched=True,
        remove_columns=["tokens", "pos_tags"]
    )

    logger.info("Grouping texts...")
    grouped = tokenized.map(
        lambda x: group_texts(x, tokenizer.model_max_length),
        batched=True
    )

    logger.info("Loading pre-trained model...")
    model = BertForMaskedLM.from_pretrained(MODEL_NAME_OR_PATH)
    state_dict = torch.load("../results/bert-large-cased/kn_erase-BR03.pt", map_location="cpu")
    model.load_state_dict(state_dict)
        
    logger.info("Running evaluation...")
    model.eval()
    all_logits = []
    all_labels = []

    for i in range(0, len(grouped), BATCH_SIZE):
        batch = grouped[i:i+BATCH_SIZE]
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        pos_tags = batch["pos_tags"]  # list of lists

        input_ids, labels = mask_tokens_with_pos(
            {"input_ids": input_ids},
            pos_tags=pos_tags,
            tokenizer=tokenizer,
            mlm_probability=0.15,
            mask_replace_prob=0.8,
            random_replace_prob=0.1,
            target_pos=16  # VERB
        )
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[1]  # outputs = (ffn_weights, tgt_logits)
            if logits is not None:
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logger.info("Computing metrics...")
    metrics = compute_metrics((all_logits, all_labels))
    print("Evaluation metrics on whole dataset:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    csv_path = "SuppressionExperimentsResults.csv"
    fieldnames = ["model", "task", "biased_rel_removed"] + list(metrics.keys())
    import os
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"model": MODEL_NAME_OR_PATH}
        row["task"] = "requirement_completion"
        row["biased_rel_removed"] = "BR03"  # Change as needed
        row.update(metrics)
        writer.writerow(row)

if __name__ == "__main__":
    main()