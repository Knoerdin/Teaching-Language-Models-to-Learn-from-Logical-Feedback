from dataclasses import dataclass
from functools import lru_cache
import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout

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


def _disable_mlflow_traces() -> None:
    os.environ.setdefault("MLFLOW_TRACE_SAMPLING_RATIO", "0.0")
    os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

    try:
        import mlflow

        mlflow.tracing.disable()
    except Exception:
        pass


@dataclass(frozen=True)
class ProverReward:
    reward: float
    status: str
    prediction: str | None = None
    state_status: str | None = None
    feedback: str | None = None
    error_message: str | None = None


@lru_cache(maxsize=1)
def _solver_dependencies():
    _disable_mlflow_traces()

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


def evaluate_correctness(
    nl_premises: str,
    nl_conclusion: str,
    formal_premises: str,
    formal_conclusion: str,
    gold_label: str,
) -> ProverReward:
    CombinedReasoningProblem, *_ = _solver_dependencies()
    solver = get_solver()

    problem = CombinedReasoningProblem(
        premises=nl_premises,
        conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
    )

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            pred, state = solver.predict(problem)
    except Exception as exc:
        captured_output = _captured_solver_output(captured_stdout, captured_stderr)
        error_message = f"{type(exc).__name__}: {exc}"
        if captured_output:
            error_message = f"{error_message}\n{captured_output}"
        return ProverReward(
            reward=-1.0,
            status="exception",
            error_message=error_message,
        )

    state_status = str(getattr(state, "status", ""))
    feedback = str(getattr(state, "feedback", "") or "")
    captured_output = _captured_solver_output(captured_stdout, captured_stderr)
    if captured_output and not feedback:
        feedback = captured_output
    if state_status.upper() == "PARSE":
        return ProverReward(
            reward=0.0,
            status="parse_error",
            prediction=str(pred),
            state_status=state_status,
            feedback=feedback,
        )

    gold = LABEL_MAP.get(gold_label.lower())
    if gold is None:
        return ProverReward(
            reward=-1.0,
            status="invalid_label",
            prediction=str(pred),
            state_status=state_status,
            feedback=feedback,
        )

    if str(pred).upper() == gold:
        return ProverReward(
            reward=1.0,
            status="correct",
            prediction=str(pred),
            state_status=state_status,
            feedback=feedback,
        )

    if str(pred).upper() == "UNKNOWN":
        return ProverReward(
            reward=0.0,
            status="unknown",
            prediction=str(pred),
            state_status=state_status,
            feedback=feedback,
        )

    return ProverReward(
        reward=-1.0,
        status="incorrect",
        prediction=str(pred),
        state_status=state_status,
        feedback=feedback,
    )


def correctness_reward(
    nl_premises: str,
    nl_conclusion: str,
    formal_premises: str,
    formal_conclusion: str,
    gold_label: str,
) -> float:
    return evaluate_correctness(
        nl_premises=nl_premises,
        nl_conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        gold_label=gold_label,
    ).reward


def _captured_solver_output(
    stdout_buffer: io.StringIO,
    stderr_buffer: io.StringIO,
) -> str:
    captured_parts = [
        stdout_buffer.getvalue().strip(),
        stderr_buffer.getvalue().strip(),
    ]
    return "\n".join(part for part in captured_parts if part)
