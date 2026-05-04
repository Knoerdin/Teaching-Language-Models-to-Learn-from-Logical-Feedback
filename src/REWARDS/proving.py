from functools import lru_cache

import hydra
from omegaconf import OmegaConf

import agent_reasoning
import agent_reasoning.utils
from agent_reasoning.pipeline import CombinedReasoningProblem


LABEL_MAP = {
    "true": "TRUE",
    "false": "FALSE",
    "uncertain": "UNKNOWN",
}


@lru_cache(maxsize=1)
def get_solver():
    agent_reasoning.utils.register_configs()
    cfg = OmegaConf.load("/path/to/agent_reasoning_rl/configs/model/ATP.yaml")
    return hydra.utils.instantiate(cfg, _convert_="partial")


def correctness_reward(
    nl_premises: str,
    nl_conclusion: str,
    formal_premises: str,
    formal_conclusion: str,
    gold_label: str,
) -> float:
    solver = get_solver()

    problem = CombinedReasoningProblem(
        premises=nl_premises,
        conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
    )

    try:
        pred, state = solver.predict(problem)
    except Exception:
        return -1.0

    if getattr(state, "status", None) == "PARSE":
        return -0.5

    gold = LABEL_MAP[gold_label.lower()]

    if str(pred).upper() == gold:
        return 1.0

    if str(pred).upper() == "UNKNOWN":
        return 0.0

    return -1.0