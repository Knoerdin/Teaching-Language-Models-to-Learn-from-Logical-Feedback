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

SFT logging uses the same MLflow fallback as GRPO: config `tracking_uri`,
then `MLFLOW_TRACKING_URI`, then the repo-local `mlruns/` directory. SFT logs
trainer metrics such as loss and eval loss through the Transformers MLflow
callback when `trainer.mlflow.enabled=true`.

Evaluate GRPO and SFT autoformalizations against the FOLIO gold FOL fields:

```bash
python src/evaluate_autoformalization.py \
  --dataset DATA/FOLIO/folio_test.jsonl \
  --model grpo=outputs/grpo_qwen3.5-9b \
  --model sft=outputs/sft_qwen3.5-9b \
  --output-dir outputs/evaluations/qwen3.5-9b
```

The main score is normalized exact-match accuracy for `premises-FOL` plus
`conclusion-FOL`; the script also reports premise-order-insensitive accuracy,
parse rate, conclusion accuracy, and premise macro F1.

When `--output-dir` is set, the evaluator also writes Markdown reports under
`OUTPUT_DIR/eval_reports/`, separated into `grpo/` and `sft/` subfolders when
the model name or path indicates the trainer type. Each model report includes
metric tables plus examples of fully correct and wrong autoformalizations. Use
`--report-dir` to choose a different report location and `--report-examples` to
change how many examples are shown per section.

On SLURM, submit and automatically tail the matching comparison job:

```bash
SLURM/submit_and_tail.sh SFT/train_sft_qwen3.5-9b.job
```
