# Teaching-Language-Models-to-Learn-from-Logical-Feedback
KI THESIS Project Nordin Jansen UvA 2026

## Training entrypoint (Hydra)

The project uses a general trainer entrypoint with model-specific Hydra configs.

- Base config: `CONFIGS/config.yaml`
- Model configs: `CONFIGS/MODELS/*.yaml`
- Trainer configs: `CONFIGS/TRAINERS/*.yaml`
- Current trainer options: `GRPO`, `SFT`

Run with defaults (GRPO):

```bash
python src/trainer.py
```

Override model config fields from CLI:

```bash
python src/trainer.py trainer.args.max_steps=500 model.model_name_or_path=Qwen/Qwen2.5-0.5B
```

Run supervised fine-tuning on the same FOLIO autoformalization prompt/completion format:

```bash
python src/trainer.py TRAINERS@trainer=SFT MODELS@model=qwen2.5
```

Run the Qwen 3.5 9B SFT comparison config:

```bash
python src/trainer.py TRAINERS@trainer=SFT_qwen3.5-9b MODELS@model=qwen3.5-9b
```

On SLURM, submit the matching comparison job from the repo root or `SLURM/`:

```bash
sbatch SLURM/train_sft_qwen3.5-9b.job
```
