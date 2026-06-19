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

The trainer treats `trainer.output_dir` as an experiment-family directory.
Every run writes checkpoints and the final adapter to a fresh run subdirectory:
on SLURM this is `trainer.output_dir/slurm-$SLURM_JOB_ID`; otherwise it uses the
Hydra timestamped run directory. The concrete run path is printed as
`Trainer output dir for this run:` at startup.

Evaluate GRPO and SFT autoformalizations against the FOLIO gold FOL fields:

```bash
python src/evaluate_autoformalization.py \
  --dataset DATA/FOLIO/folio_test.jsonl \
  --model grpo=outputs/grpo_qwen3.5-9b/slurm-<grpo-job-id> \
  --model sft=outputs/sft_qwen3.5-9b/slurm-<sft-job-id> \
  --output-dir outputs/evaluations/qwen3.5-9b
```

By default the evaluator uses an autonomous prompt: the model sees only the
natural-language premises and conclusion, not the per-example gold predicate
inventory. The main formalization score is normalized exact-match accuracy for
`premises-FOL` plus `conclusion-FOL`; the script also reports
premise-order-insensitive accuracy, parse rate, conclusion accuracy, premise
macro F1, schema predicate/constant F1, and optional prover label accuracy.
Use `--include-gold-schema` only to reproduce the older schema-conditioned
setup. Use `--postprocess` only for diagnostic cleanup, because it canonicalizes
against gold FOL symbols.

To run a Draft-and-Prune style evaluation, sample multiple natural-language
translation plans, generate one deterministic FOL candidate from each plan,
prune candidates that the theorem prover cannot execute, and majority-vote the
surviving prover labels:

```bash
python src/evaluate_autoformalization.py \
  --dataset DATA/FOLIO/folio_test.jsonl \
  --model grpo=outputs/grpo_qwen3.5-9b/slurm-<grpo-job-id> \
  --draft-and-prune \
  --paths 20 \
  --repair-rounds 2 \
  --run-prover \
  --output-dir outputs/evaluations/qwen3.5-9b-dp
```

Draft-and-Prune reports `dp_label_accuracy`, execution rate, path accuracy,
hit rate, abstention rate, tie rate, and the vote counts/path summaries in the
prediction JSONL. Invalid or unparseable FOL paths are counted as pruned, not as
unknown labels.
The evaluator also logs per-checkpoint elapsed time plus the latest D&P path
status counts. If repair rounds dominate a run, use `--repair-statuses` to
limit which failed paths are regenerated, for example `--repair-statuses
parse_error` to skip raw `not_parsed` outputs or `--repair-statuses none` to
prune all failed paths immediately.

On Snellius, run D&P evaluation through SLURM instead of the login node:

```bash
SLURM/submit_and_tail.sh EVAL/evaluate_dp_qwen3.5-9b_smoke.job
SLURM/submit_and_tail.sh EVAL/evaluate_dp_qwen3.5-9b.job
```

Both jobs default to `outputs/grpo_qwen3.5-9b/checkpoint-3000`. Override the
model from `sbatch` when needed, for example:

```bash
sbatch --export=ALL,MODEL_NAME=sft,MODEL_PATH=outputs/sft_qwen3.5-9b/checkpoint-3000 \
  SLURM/EVAL/evaluate_dp_qwen3.5-9b_smoke.job
```

The full D&P SLURM job also accepts `PATHS`, `REPAIR_ROUNDS`,
`REPAIR_STATUSES`, `BATCH_SIZE`, and `LIMIT` through `sbatch --export`.

To compute a cheap single-path baseline from existing D&P predictions without
calling the model again, score one saved path per example:

```bash
PYTHONPATH=src python src/evaluate_saved_dp_paths.py \
  --predictions sft_path1=outputs/evaluations/qwen3.5-9b-dp/sft_predictions.jsonl \
  --predictions grpo_final_path1=outputs/evaluations/grpo_final_20260531_dp/grpo_final_20260531_predictions.jsonl \
  --output-dir outputs/evaluations/offline_single_path_from_dp/path1
```

This measures a single saved D&P path without majority voting. It does not
reconstruct a true direct-prompt non-D&P generation, because those generations
were not produced by the D&P run.

To evaluate only the metrics used in the paper-style per-model plots without
Draft-and-Prune, run the lightweight direct evaluator:

```bash
PYTHONPATH=src python src/evaluate_plotted_metrics.py \
  --dataset DATA/FOLIO/folio_test.jsonl \
  --model base=Qwen/Qwen3.5-9B \
  --model sft=outputs/sft_qwen3.5-9b/checkpoint-1000 \
  --model grpo_final=outputs/grpo_qwen3.5-9b/final_20260531_1521_grpo/checkpoint-2000 \
  --max-new-tokens 192 \
  --repetition-penalty 1.1 \
  --output-dir outputs/evaluations/plotted_metrics
```

This evaluator does one direct generation per example and always runs the prover
for label accuracy/F1. It does not sample D&P paths, run repairs, or report D&P
search-shape metrics. In these plotted metrics, `parse_rate` is the solver/FOL
parser parse rate, not the text-section extraction rate. The old extraction-only
number is still written as `format_extraction_rate` in the metrics JSON. The
evaluator also writes readable per-example generations under
`OUTPUT_DIR/model_outputs/`, including the raw model output, extracted FOL, gold
FOL, prover status, and label decision. To plot those outputs, use:

```bash
PYTHONPATH=src python src/plot_evaluation_metrics.py \
  --prediction "Qwen3.5-9B base=outputs/evaluations/plotted_metrics/base_predictions.jsonl" \
  --prediction "SFT step 1000=outputs/evaluations/plotted_metrics/sft_predictions.jsonl" \
  --prediction "GRPO final=outputs/evaluations/plotted_metrics/grpo_final_predictions.jsonl" \
  --output-dir outputs/evaluations/comparison_plots
```

Each model gets its own plot. The plot intentionally omits D&P search-shape
metrics and shows only label accuracy, solver/FOL parser parse rate, gold-FOL
exact accuracy, and label F1 scores.

On Snellius, run the same direct final evaluation through SLURM:

```bash
SLURM/submit_and_tail.sh \
  --export=ALL,EVAL_VENV=/path/to/eval/env,VAMPIRE_BIN=/path/to/vampire,AGENT_REASONING_SRC=/path/to/agent_reasoning_rl/src \
  EVAL/evaluate_plotted_metrics_qwen3.5-9b_smoke.job

SLURM/submit_and_tail.sh \
  --export=ALL,EVAL_VENV=/path/to/eval/env,VAMPIRE_BIN=/path/to/vampire,AGENT_REASONING_SRC=/path/to/agent_reasoning_rl/src \
  EVAL/evaluate_plotted_metrics_qwen3.5-9b.job
```

These jobs write metrics, predictions, plots, and a runtime log under
`outputs/final_eval_runs/<run-name>/`. `EVAL_VENV` should point to a Python
3.12 evaluation environment; if it is omitted, the job tries `.venv_eval`,
`eval_venv`, then `.venv`. The plotted-metrics jobs include the base
`Qwen/Qwen3.5-9B` model by default; set `INCLUDE_BASE_MODEL=0` to evaluate only
SFT and GRPO, or set `BASE_MODEL_PATH` to a local cached/exported base model.
The full plotted-metrics job defaults to `MAX_NEW_TOKENS=192` and
`REPETITION_PENALTY=1.1`, matching the Qwen3.5 GRPO generation settings more
closely than the older 256-token greedy run. Set `WRITE_MODEL_OUTPUTS=0` to skip
the readable output files, or `MODEL_OUTPUT_EXAMPLES=N` to write only the first
`N` examples per model.
Override paths or labels from `sbatch` when needed, for example:

```bash
sbatch --export=ALL,RUN_NAME=final_eval_qwen35,EVAL_VENV=/path/to/eval/env,BASE_MODEL_PATH=Qwen/Qwen3.5-9B,SFT_MODEL_PATH=outputs/sft_qwen3.5-9b/checkpoint-1000 \
  SLURM/EVAL/evaluate_plotted_metrics_qwen3.5-9b.job
```

On SLURM, submit and automatically tail the matching comparison job:

```bash
SLURM/submit_and_tail.sh SFT/train_sft_qwen3.5-9b.job
```

The submit helper writes SLURM stdout/stderr under `SLURM/logs/SFT/` or
`SLURM/logs/GRPO/`, grouped by job name and resource settings. If submitting
directly with `sbatch`, run it from the repo root so the job-file fallback paths
also resolve under `SLURM/logs/`.
