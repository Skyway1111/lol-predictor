"""Run an exporter in the calculator checkout and validate its JSON result."""

import argparse
import json
import subprocess
from pathlib import Path


def dump_from_calculator(calculator: Path, interpreter: Path, program: str) -> dict:
    """Run the calculator's own registry and return its control declarations.

    Raises:
        RuntimeError: the checkout, its interpreter, or its answer is missing.
    """
    if not (calculator / "src" / "calculator").is_dir():
        raise RuntimeError(
            f"{calculator} is not a merged-calculator checkout (no src/calculator/). "
            "Pass --calculator <path to the sibling checkout>."
        )
    if not interpreter.is_file():
        raise RuntimeError(
            f"{interpreter} does not exist — the calculator's own interpreter runs the "
            "dump. Pass --python <path>, or create the checkout's .venv."
        )
    finished = subprocess.run(  # noqa: S603 - both argv entries are checked paths above
        [str(interpreter), "-c", program],
        cwd=calculator,
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            f"the calculator dump exited {finished.returncode}:\n{finished.stderr[-2000:]}"
        )
    payload = json.loads(finished.stdout)
    if not payload.get("champions"):
        raise RuntimeError("the calculator dump named no champions")
    return payload


def read_cli_dump(program: str) -> tuple[dict, Path]:
    """Read a reviewed-artifact export using the calculator's configured interpreter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculator", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="The calculator's interpreter; defaults to its .venv/Scripts/python.exe.",
    )
    args = parser.parse_args()
    calculator = args.calculator.resolve()
    interpreter = args.python or calculator / ".venv" / "Scripts" / "python.exe"
    return dump_from_calculator(calculator, interpreter, program), calculator
