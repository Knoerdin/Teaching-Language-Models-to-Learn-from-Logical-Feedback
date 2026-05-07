from functools import lru_cache
import warnings

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except ModuleNotFoundError:
    LangChainPendingDeprecationWarning = Warning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


LABEL_MAP = {
    "true": "TRUE",
    "false": "FALSE",
    "uncertain": "UNKNOWN",
}


class SolverUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _solver_dependencies():
    try:
        from agent_reasoning.pipeline import CombinedReasoningProblem
        from agent_reasoning.symbolic_solver import TheoremProverPipeline
        from agent_reasoning.symbolic_solver.fol import TPTPParser, FOLLinter
        from agent_reasoning.symbolic_solver.fol.tools import VampireReasoner
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "unknown package"
        raise SolverUnavailableError(
            "FOL solver dependencies are not installed. "
            "This project now requires Python 3.12 and agent-reasoning. "
            "Create/use a Python 3.12 environment and run `python -m pip install -e .`. "
            "If you want the sibling checkout instead, run "
            "`python -m pip install -e ../agent_reasoning_rl` in that same environment. "
            f"Missing import: {missing_package}."
        ) from exc

    return (
        CombinedReasoningProblem,
        TheoremProverPipeline,
        TPTPParser,
        FOLLinter,
        VampireReasoner,
    )


@lru_cache(maxsize=1)
def get_solver():
    (
        _,
        TheoremProverPipeline,
        TPTPParser,
        FOLLinter,
        VampireReasoner,
    ) = _solver_dependencies()

    parser = TPTPParser(convert=True, feedback_msg=True)
    solver = VampireReasoner()
    linter = FOLLinter()
    tests = []

    return TheoremProverPipeline(
        parser=parser,
        solver=solver,
        linter=linter,
        tests=tests,
        inconsistent_as_error=True,
    )


def correctness_reward(
    nl_premises: str,
    nl_conclusion: str,
    formal_premises: str,
    formal_conclusion: str,
    gold_label: str,
) -> float:
    CombinedReasoningProblem, *_ = _solver_dependencies()
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

    gold = LABEL_MAP.get(gold_label.lower())
    if gold is None:
        return -1.0

    if str(pred).upper() == gold:
        return 1.0

    if str(pred).upper() == "UNKNOWN":
        return 0.0

    return -1.0
