import argparse
import inspect
import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from laststep_data_utils import build_cls_examples, build_laststep_samples, concat_history_frames, load_checkin_frame, save_json


def find_last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    ckpts = []
    for name in os.listdir(output_dir):
        m = re.match(r"^checkpoint-(\d+)$", name)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(output_dir, name)))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    pred1 = np.argmax(logits, axis=1)
    acc1 = float((pred1 == labels).mean())
    top3 = np.argsort(-logits, axis=1)[:, :3]
    acc3 = float((top3 == labels[:, None]).any(axis=1).mean())
    return {"acc1": acc1, "acc3": acc3}


def safe_set_pad_token(model, pad_token_id: int):
    model.config.pad_token_id = pad_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.pad_token_id = pad_token_id
        except Exception:
            pass


class FP32CELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss()(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


class ManualEvalSaveBestCallback(TrainerCallback):
    def __init__(self, best_dir: str, tokenizer, eval_steps: int):
        self.best_dir = best_dir
        self.tokenizer = tokenizer
        self.eval_steps = int(eval_steps)
        self.best = -1.0
        self.trainer = None
        self._in_eval = False
        os.makedirs(self.best_dir, exist_ok=True)

    def _maybe_save(self, metrics, state):
        cur = None if not metrics else metrics.get("eval_acc1")
        if cur is None:
            return
        cur = float(cur)
        if cur <= self.best:
            return
        self.best = cur
        self.trainer.model.save_pretrained(self.best_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.best_dir)
        save_json(
            os.path.join(self.best_dir, "best_metric.json"),
            {"best_eval_acc1": self.best, "global_step": int(state.global_step)},
        )
        print(f"\n[BestModel] eval_acc1={self.best:.6f} saved to {self.best_dir}\n")

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or self._in_eval or self.eval_steps <= 0:
            return control
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._in_eval = True
            metrics = self.trainer.evaluate()
            self._in_eval = False
            self._maybe_save(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer is None or self._in_eval:
            return control
        self._in_eval = True
        metrics = self.trainer.evaluate()
        self._in_eval = False
        self._maybe_save(metrics, state)
        return control


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("BASE_MODEL", ""))
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--history_limit", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--use_4bit", action="store_true", default=True)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", action="store_true", default=False)
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.model_dir:
        raise ValueError("Please set --model_dir or export BASE_MODEL.")
    if hasattr(torch.autograd, "graph") and hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    os.makedirs(args.out_dir, exist_ok=True)
    best_dir = os.path.join(args.out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    train_df = load_checkin_frame(args.train_csv, "train")
    val_df = load_checkin_frame(args.val_csv, "val")
    val_history_df = concat_history_frames([train_df, val_df])

    cats = sorted(train_df["POI_catname"].astype(str).unique().tolist())
    label2id = {c: i for i, c in enumerate(cats)}
    id2label = {i: c for i, c in enumerate(cats)}

    train_samples, train_stats = build_laststep_samples(
        train_df,
        train_df,
        history_limit=args.history_limit,
        require_history=True,
    )
    val_samples, val_stats = build_laststep_samples(
        val_df,
        val_history_df,
        history_limit=args.history_limit,
        require_history=True,
    )

    train_ex, dropped_tr = build_cls_examples(train_samples, label2id)
    val_ex, dropped_va = build_cls_examples(val_samples, label2id)
    save_json(
        os.path.join(args.out_dir, "sample_stats.json"),
        {
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_examples": len(train_ex),
            "val_examples": len(val_ex),
            "dropped_train_examples": dropped_tr,
            "dropped_val_examples": dropped_va,
            "history_limit": args.history_limit,
        },
    )

    train_ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(ex):
        t = tokenizer(ex["text"], truncation=True, max_length=args.max_len)
        t["labels"] = ex["labels"]
        return t

    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_fn, remove_columns=val_ds.column_names)

    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features):
        batch = base_collator(features)
        batch["labels"] = batch["labels"].long()
        return batch

    bnb = None
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        dtype=dtype,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.problem_type = "single_label_classification"
    safe_set_pad_token(model, tokenizer.pad_token_id)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.batch_size * args.grad_accum))
    total_train_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_train_steps * 0.03))

    train_kwargs = {
        "output_dir": args.out_dir,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_steps": warmup_steps,
        "max_grad_norm": 0.3,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "do_eval": True,
        "report_to": "none",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "paged_adamw_8bit",
        "remove_unused_columns": False,
    }
    ta_sig = inspect.signature(TrainingArguments.__init__)
    train_kwargs = {k: v for k, v in train_kwargs.items() if k in ta_sig.parameters}
    train_args = TrainingArguments(**train_kwargs)

    best_cb = ManualEvalSaveBestCallback(best_dir=best_dir, tokenizer=tokenizer, eval_steps=args.eval_steps)
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [best_cb],
    }
    trainer_sig = inspect.signature(FP32CELossTrainer.__init__)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    trainer = FP32CELossTrainer(**trainer_kwargs)
    best_cb.trainer = trainer

    last_ckpt = find_last_checkpoint(args.out_dir)
    if last_ckpt:
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    label_space = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    save_json(os.path.join(args.out_dir, "label_space.json"), label_space)

    final_dir = os.path.join(args.out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
