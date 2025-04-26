#!/usr/bin/env python3
"""
finetune_qwen25vl7b.py
──────────────────────
Fine‑tune Qwen‑2.5‑VL‑7B‑Instruct (Hugging Face) on a local JSONL+image dataset.

Requires:
  pip install -q "git+https://github.com/huggingface/transformers" \
                 accelerate peft bitsandbytes qwen-vl-utils[decord]==0.0.8 \
                 lightning nltk
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, List, Tuple

import lightning as L
import torch
from nltk import edit_distance
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
os.environ["HF_HOME"] = "/workspace/huggingface"
# or
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_models"


# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_MESSAGE = Path(
    '/workspace/dslab25/training/qwen/data/system_message.txt'
).with_name("system_message.txt").read_text(encoding="utf-8")
# ────────────────────────────────────────────────────────────────────────────────


def format_chat(image_dir: Path, entry: dict) -> List[dict]:
    """Return a 3‑turn conversation in the format Qwen expects."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_dir / entry["image"])},
                {"type": "text", "text": entry["prefix"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": entry["suffix"]}]},
    ]


class JSONLDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_dir: Path):
        self.image_dir = image_dir
        self.entries = [json.loads(l) for l in jsonl_path.read_text().splitlines()]

    def __len__(self):  # noqa: D401
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Any, dict, List[dict]]:  # noqa: D401
        entry = self.entries[idx]
        return None, entry, format_chat(self.image_dir, entry)


# ─── Collate fns ────────────────────────────────────────────────────────────────
def make_collate(processor):
    def train_collate(batch):
        _, _, examples = zip(*batch)
        texts = [processor.apply_chat_template(e, tokenize=False) for e in examples]
        imgs = [process_vision_info(e)[0] for e in examples]
        model_in = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        labels = model_in["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tkn in (151652, 151653, 151655):
            labels[labels == tkn] = -100
        return (
            model_in["input_ids"],
            model_in["attention_mask"],
            model_in["pixel_values"],
            model_in["image_grid_thw"],
            labels,
        )

    def eval_collate(batch):
        _, data, examples = zip(*batch)
        suffixes = [d["suffix"] for d in data]
        prompts = [processor.apply_chat_template(e[:2], tokenize=False) for e in examples]
        imgs = [process_vision_info(e[:2])[0] for e in examples]
        model_in = processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
        return (
            model_in["input_ids"],
            model_in["attention_mask"],
            model_in["pixel_values"],
            model_in["image_grid_thw"],
            suffixes,
        )

    return train_collate, eval_collate


# ─── Lightning module ───────────────────────────────────────────────────────────
class QwenTrainer(L.LightningModule):
    def __init__(self, cfg, model, processor, train_set, val_set):
        super().__init__()
        self.save_hyperparameters()
        self.cfg, self.model, self.processor = cfg, model, processor
        self.train_set, self.val_set = train_set, val_set
        self.train_collate, self.eval_collate = make_collate(processor)

    # ╭─ training ╮
    def training_step(self, batch, _):
        ids, msk, pix, thw, lbl = batch
        loss = self.model(
            input_ids=ids,
            attention_mask=msk,
            pixel_values=pix,
            image_grid_thw=thw,
            labels=lbl,
        ).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ╭─ validation ╮
    def validation_step(self, batch, _):
        ids, msk, pix, thw, refs = batch
        gen_ids = self.model.generate(
            input_ids=ids,
            attention_mask=msk,
            pixel_values=pix,
            image_grid_thw=thw,
            max_new_tokens=256,
        )
        outs = self.processor.batch_decode(
            [o[len(i) :] for i, o in zip(ids, gen_ids)],
            skip_special_tokens=True,
        )
        score = sum(edit_distance(o, r) / max(len(o), len(r)) for o, r in zip(outs, refs))
        self.log("val_edit_dist", score / len(refs), prog_bar=True)

    # ╭─ loaders ╮
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=self.train_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=self.eval_collate,
        )

    # ╭─ optim ╮
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.cfg["lr"])


# ─── Entry‑point ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=Path)
    ap.add_argument("--epochs", default=6, type=int)
    ap.add_argument("--save_dir", default="qwen_2_5_vl_7b_ft", type=Path)
    args = ap.parse_args()

    train_jsonl = args.data_root / "train" / "annotations.jsonl"
    val_jsonl = args.data_root / "val" / "annotations.jsonl"

    train_set = JSONLDataset(train_jsonl, args.data_root / "train")
    val_set = JSONLDataset(val_jsonl, args.data_root / "val")

    # ─── Model + processor ────────────────────────────────────────────────────
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj"]
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", quantization_config=bnb_cfg, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, lora_cfg)
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    cfg = dict(batch_size=1, lr=2e-4)
    lit = QwenTrainer(cfg, model, processor, train_set, val_set)

    # ─── Checkpoint callback ──────────────────────────────────────────────────
    class SaveBoth(L.Callback):
        def __init__(self, out: Path):  # noqa: D401
            self.out = out

        def on_train_epoch_end(self, trainer, pl_module):
            path = self.out / f"epoch_{trainer.current_epoch}"
            path.mkdir(parents=True, exist_ok=True)
            pl_module.processor.save_pretrained(path)
            pl_module.model.save_pretrained(path)
            print(f"[ckpt] {path}")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[SaveBoth(args.save_dir)],
        precision="bf16-mixed",
    )

    trainer.fit(lit)


if __name__ == "__main__":
    main()