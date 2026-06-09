"""Prompt construction vendored from GAIR-NLP/AIME-Preview eval/eval.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_TYPE = "qwen-instruct"
DEFAULT_DATA_NAME = "math"


def get_three_prompt(prompt_type: str, data_name: str) -> tuple[str, str, str]:
    file_path = PROMPTS_DIR / prompt_type / f"{data_name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "system_prompt"):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")

    if hasattr(module, "few_shot_prompt"):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")

    if hasattr(module, "question_format"):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def build_messages(
    question: str,
    prompt_type: str = DEFAULT_PROMPT_TYPE,
    data_name: str = DEFAULT_DATA_NAME,
    use_few_shot: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages using AIME-Preview's qwen-instruct + surround_with_messages format."""
    system_prompt, few_shot_prompt, question_format = get_three_prompt(prompt_type, data_name)

    if use_few_shot:
        user_content = few_shot_prompt + question_format.format(question=question)
    else:
        user_content = question_format.format(question=question)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
