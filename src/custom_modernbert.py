# custom_modernbert.py
import os, importlib.util
from typing import Tuple
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM

def _dynamic_import_modernbert(local_dir: str):
    config_py = modeling_py = None
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.startswith("configuration_") and "modernbert" in f:
                config_py = os.path.join(root, f)
            if f.startswith("modeling_") and "modernbert" in f:
                modeling_py = os.path.join(root, f)
    if not (config_py and modeling_py):
        raise RuntimeError("ModernBERT config/modeling files not found in snapshot.")

    def _load(path: str):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    conf_mod = _load(config_py)
    mdl_mod  = _load(modeling_py)
    ModernBertConfig = getattr(conf_mod, "ModernBertConfig")
    ModernBertForMaskedLM = getattr(mdl_mod, "ModernBertForMaskedLM")
    return ModernBertConfig, ModernBertForMaskedLM

class ModernBertForMaskedLM(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.config = inner.config
        # TODO: keep your hook discovery & forward logic here (unchanged)

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kw):
        # 1) Fast path: Auto* with remote code
        try:
            cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
            mdl = AutoModelForMaskedLM.from_pretrained(
                name_or_path, config=cfg, trust_remote_code=True, **kw
            )
            if getattr(mdl.config, "model_type", "").lower() != "modernbert":
                raise RuntimeError(f"Loaded model_type={mdl.config.model_type!r}, expected 'modernbert'")
            return cls(mdl)
        except Exception as e:
            print(f"[ModernBERT] Auto* load failed: {e}\nFalling back to snapshot import…")

        # 2) Fallback: snapshot + direct import of classes, then load weights locally
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=name_or_path,
            allow_patterns=[
                "config.json",
                "configuration_*modernbert*.py",
                "modeling_*modernbert*.py",
                "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
                "*.safetensors",
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "merges.txt",
            ],
        )
        ModernBertConfig, ModernBertForMaskedLM_real = _dynamic_import_modernbert(local_dir)
        base_cfg = ModernBertConfig.from_pretrained(local_dir)
        inner = ModernBertForMaskedLM_real.from_pretrained(local_dir, config=base_cfg)
        return cls(inner)
