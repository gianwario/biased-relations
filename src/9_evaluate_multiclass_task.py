# -------------- evaluate_multiclass_task.py --------------
import os, sys, glob, csv, time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings("ignore")

# Use your metrics exactly as defined by the authors
sys.path.append("../")
from utils.evaluation import compute_metrics_hf

# ======== CONFIG (edit these) ========
DATASETS_BASE_PATH = "./datasets/preprocessed"      # folder containing <DATASET_NAME>.parquet
SPLITS_BASE_PATH   = "./datasets/splits"            # folder containing <DATASET_NAME>.k-fold.*.csv
DATASET_NAME       = "sentiment"                   # <-- change to your dataset name

PARQUET_PATH       = f"{DATASETS_BASE_PATH}/{DATASET_NAME}.parquet"
SPLITS_GLOB        = f"{SPLITS_BASE_PATH}/{DATASET_NAME}.k-fold.*.csv"

MODEL_LIST = [
    "aieng-lab/bert-large-cased_sentiment",  # fine-tuned multiclass model
    "bert-large-cased",                       # raw model (sanity)
]

BR_LIST = ["None", "BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]

CHECKPOINT_ROOT = "../results"  # expects ../results/<model_dirname>/kn_erase-<BR>.pt

MAX_LEN     = 128
BATCH_SIZE  = 32
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV  = f"multiclass_{DATASET_NAME}_suppression_results.csv"
# =====================================


# ---------- helpers ----------
def model_dirname(model_name: str) -> str:
    return model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name

def ckpt_path(model_name: str, br: str) -> str:
    return os.path.join(CHECKPOINT_ROOT, model_dirname(model_name), f"kn_erase-{br}.pt")

def flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Make compute_metrics_hf output CSV-friendly."""
    flat: Dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, dict) and "accuracy" in v and len(v) == 1:
            flat["accuracy"] = v["accuracy"]
        elif isinstance(v, dict) and "confusion_matrix" in v:
            cm = v["confusion_matrix"]
            # try to flatten a square CM if small
            try:
                n = len(cm)
                for i in range(n):
                    for j in range(n):
                        flat[f"cm_{i}{j}"] = cm[i][j]
            except Exception:
                flat["confusion_matrix"] = str(cm)
        else:
            flat[k] = v
    return flat


# ---------- Data ----------
def load_test_df() -> Tuple[pd.DataFrame, str, int]:
    df = pd.read_parquet(PARQUET_PATH)

    # text column
    if "text_clean" in df.columns:
        df["text_clean"] = df["text_clean"].astype(str).str.replace("</s>", "", regex=False)
        text_col = "text_clean"
    elif "text" in df.columns:
        df["text"] = df["text"].astype(str).str.replace("</s>", "", regex=False)
        text_col = "text"
        df = df.rename(columns={"text": "text_clean"})
    else:
        raise KeyError("Expected a 'text_clean' or 'text' column in the parquet.")

    # label column
    if "label" in df.columns:
        label_col = "label"
    elif "labels" in df.columns:
        label_col = "labels"
        df = df.rename(columns={"labels": "label"})
    else:
        raise KeyError("Expected a 'label' (or 'labels') column for multiclass.")

    # ensure integer classes
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    if df["label"].isna().any():
        raise ValueError("Found NaNs in label column after coercion.")

    # load splits
    matches = glob.glob(SPLITS_GLOB)
    if not matches:
        raise FileNotFoundError(f"No splits CSV matches: {SPLITS_GLOB}")
    splits = pd.read_csv(matches[0])
    if "fold_1" not in splits.columns:
        raise KeyError("Splits CSV must contain a 'fold_1' column.")

    test_df = df.loc[splits["fold_1"] == 2].copy()

    # class count
    num_labels = int(test_df["label"].nunique())
    print(f"Detected num_labels={num_labels}. Label distribution: {test_df['label'].value_counts().sort_index().to_dict()}")

    return test_df, "text_clean", num_labels


class MultiClassDS(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, tok, max_len: int):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.tok = tok
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            str(self.df.loc[i, self.text_col]),
            truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.df.loc[i, "label"]), dtype=torch.long)
        return item


# ---------- BR erasure partial-load ----------
def load_erasure_into_encoder(model: AutoModelForSequenceClassification, path: str) -> int:
    sd_src = torch.load(path, map_location="cpu")
    sd_tgt = model.state_dict()
    to_load = {k: v for k, v in sd_src.items() if (k in sd_tgt and sd_tgt[k].shape == v.shape)}
    model.load_state_dict(to_load, strict=False)
    return len(to_load)


# ---------- Eval loop ----------
@torch.no_grad()
def collect_logits_labels(model: AutoModelForSequenceClassification, loader: DataLoader
                          ) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(
            **{k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids"] if k in batch},
            labels=batch["labels"]
        )
        logits_all.append(out.logits.detach().cpu().numpy())   # (B, C)
        labels_all.append(batch["labels"].detach().cpu().numpy())  # (B,)
    return np.concatenate(logits_all, 0), np.concatenate(labels_all, 0)


def load_model_preserve_head(model_name: str, dataset_num_labels: int) -> AutoModelForSequenceClassification:
    """
    1) Try to load as-is (keeps finetuned head if present).
    2) If head size != dataset classes, reload with num_labels=dataset_num_labels (random head).
    """
    m = AutoModelForSequenceClassification.from_pretrained(model_name)
    if getattr(m.config, "num_labels", None) != dataset_num_labels:
        # Re-init head to correct size for raw/base checkpoints
        del m
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=dataset_num_labels,
            problem_type="single_label_classification"
        )
        print(f"⚠️ {model_name}: reset classifier head to num_labels={dataset_num_labels}.")
    return m


def main():
    test_df, text_col, num_labels = load_test_df()

    wrote_header = False
    fieldnames: List[str] = []

    for model_name in MODEL_LIST:
        tok = AutoTokenizer.from_pretrained(model_name if "bert" not in model_name else "bert-large-cased")
        if tok.pad_token is None and hasattr(tok, "eos_token"):
            tok.pad_token = tok.eos_token

        ds = MultiClassDS(test_df, text_col, tok, MAX_LEN)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        for br in BR_LIST:
            # fresh model per run
            model = load_model_preserve_head(model_name, num_labels).to(DEVICE)

            if br != "None":
                path = ckpt_path(model_name, br)
                if not os.path.exists(path):
                    print(f"[SKIP] missing erasure checkpoint: {path}")
                    continue
                loaded = load_erasure_into_encoder(model, path)
                print(f"[{model_name} | {br}] loaded_tensors={loaded}")
                if loaded == 0:
                    print("⚠️ erasure loaded 0 overlapping tensors; results may not reflect erasure.")

            tic = time.perf_counter()
            logits, labels = collect_logits_labels(model, dl)
            # your metrics (supports multiclass)
            metrics = flatten_metrics(compute_metrics_hf((logits, labels)))
            toc = time.perf_counter()

            row = {
                "model": model_name,
                "task": DATASET_NAME,
                "biased_rel_removed": br,
                "time": round(toc - tic, 3),
                **metrics
            }

            if not wrote_header:
                fieldnames = list(row.keys())
                with open(OUTPUT_CSV, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
                wrote_header = True

            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(f"✓ {model_name} | {br} | "
                  f"acc={row.get('accuracy', float('nan')):.4f} "
                  f"f1_macro={row.get('f1_macro', float('nan')):.4f} "
                  f"time={row['time']}s")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done.")

if __name__ == "__main__":
    main()
# -----------------------------------------------------------
