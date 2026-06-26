# Training Pipeline (QLoRA)

## Overview

The trainer fine-tunes the base model with **QLoRA**: the base is loaded in
4-bit, frozen, and a small **LoRA** adapter is trained on top. This makes
fine-tuning a 7B model feasible on a single Colab GPU.

Entry point: `scripts/train.py` → `src.training.trainer.run_training()`.

## Stages

1. **Seed & setup.** Global seeding for reproducibility; directories ensured.
2. **Load base model (4-bit).** `load_model_and_tokenizer(for_training=True)`:
   - `BitsAndBytesConfig` with `nf4`, double quantisation, bf16 compute dtype;
   - `prepare_model_for_kbit_training` (enables input grads, disables KV cache);
   - tokenizer padded on the **right** for training.
3. **Attach LoRA.** `get_peft_model` with the configured rank/alpha/dropout on
   the Qwen projection layers (`q,k,v,o,gate,up,down`). Trainable-parameter ratio
   is logged.
4. **Prepare data.** `prepare_dataset` normalises any supported format, renders
   each example through the chat template into a `text` column, and splits
   train/eval.
5. **Build `SFTTrainer`.** `SFTConfig` is assembled from `TrainingConfig` and
   filtered to the installed library's accepted fields.
6. **Train.** With gradient accumulation/checkpointing, bf16 (auto), periodic
   evaluation, TensorBoard logging and bounded checkpoint retention.
7. **Save.** The final adapter, tokenizer and a JSON config snapshot are written
   to `adapters/<run>`.

## Key configuration (`config/training_config.py`)

| Field | Default | Meaning |
|-------|---------|---------|
| `run_name` | `finai-qlora` | Names the checkpoint/adapter/TensorBoard dirs |
| `per_device_train_batch_size` | 2 | Micro-batch per step |
| `gradient_accumulation_steps` | 8 | → effective batch size 16 |
| `learning_rate` | 2e-4 | Typical for LoRA |
| `lr_scheduler_type` | cosine | With `warmup_ratio` 0.03 |
| `num_train_epochs` / `max_steps` | 3 / -1 | `max_steps>0` overrides epochs |
| `max_seq_length` | 2048 | Truncation length |
| `gradient_checkpointing` | True | Trades compute for memory |
| `save_total_limit` | 3 | Keep only the newest 3 checkpoints |
| `eval_split_ratio` | 0.05 | Held-out eval fraction (0 disables) |
| `optim` | paged_adamw_8bit | Memory-efficient optimiser |
| `report_to` | tensorboard | Logging backend |
| LoRA `r/alpha/dropout` | 16/32/0.05 | Adapter capacity |

All are overridable by CLI flag or environment variable.

## Precision & attention (auto-detected)

- **bf16** is used on Ampere+ GPUs (A100/L4); **fp16** on older CUDA GPUs (T4);
  both off on CPU. Override with the `bf16`/`fp16` config fields.
- **flash-attention-2** is used only when the `flash_attn` package is installed
  *and* the GPU supports it; otherwise PyTorch **SDPA**.

## CLI examples

```bash
# Smoke test (50 steps, no eval) on bundled data:
python scripts/train.py --run-name finai-qlora \
    --dataset data/samples/openai_sample.jsonl --max-steps 50 --eval-ratio 0

# Full run on your own data:
python scripts/train.py --run-name finai-v1 --dataset /path/data.jsonl \
    --epochs 3 --train-batch-size 2 --grad-accum 16 --max-seq-len 2048

# Tune the adapter:
python scripts/train.py --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
```

## Resuming

```bash
python scripts/train.py --run-name finai-v1 --resume
```

`--resume` calls `get_last_checkpoint(checkpoints/<run>)`; if a checkpoint
exists, optimiser/scheduler/step state are restored. If none is found, a warning
is logged and training starts fresh.

## Monitoring with TensorBoard

```bash
tensorboard --logdir "$FINAI_HOME/runs/<run-name>"
```

On Colab: `%load_ext tensorboard` then `%tensorboard --logdir <path>`.

## Version tolerance

TRL/Transformers rename constructor arguments between releases (e.g.
`evaluation_strategy` → `eval_strategy`, and the SFT max-length field). The
trainer:

- inspects the installed `SFTConfig`/`SFTTrainer` signatures,
- drops arguments the installed version does not accept,
- remaps known renames via `_CONFIG_ALIASES`,
- detects `tokenizer` vs `processing_class`.

If a future release renames another argument, add the mapping to
`_CONFIG_ALIASES` in `src/training/trainer.py`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM | Lower `--train-batch-size`, raise `--grad-accum`, reduce `--max-seq-len`. |
| `tokenizer has no chat template` | Base model + chat data; use an `-Instruct` model or plain-text data. |
| Eval never runs | `eval_split_ratio` is 0 or dataset too small (<10 examples). |
| Loss is NaN | Try fp16 off / bf16 on (newer GPU), or lower the learning rate. |
