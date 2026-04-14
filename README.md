# Teaching-Language-Models-to-Learn-from-Logical-Feedback
KI THESIS Project Nordin Jansen UvA 2026

## Training entrypoint (Hydra)

The project now uses a general trainer entrypoint with model-specific Hydra configs.

- Base config: `CONFIGS/config.yaml`
- Model configs: `CONFIGS/MODELS/*.yaml`
- Current model option: `GRPO`

Run with defaults (GRPO):

```bash
python src/trainer.py
```

Override model config fields from CLI:

```bash
python src/trainer.py model.training.max_steps=500 model.model_name_or_path=Qwen/Qwen2.5-0.5B
```

When you add a new model config, also add a trainer function and register it in `src/trainer.py`.
