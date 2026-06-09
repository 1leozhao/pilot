"""extract_answer vendored from GAIR-NLP/AIME-Preview eval/utils/parser.py."""

from __future__ import annotations

import re


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    pred_str = pred_str.replace("\u043a\u0438", "")

    pred = ""

    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    return pred


def _humanize_latex(text: str) -> str:
    """Convert common LaTeX fragments to plain readable text."""
    s = text.replace("\\\\", "\\")

    for _ in range(3):
        s = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            lambda m: f"{m.group(1)}/{m.group(2)}",
            s,
        )

    s = re.sub(r"\\(?:boxed|text)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\left[\(\[\{]", "(", s)
    s = re.sub(r"\\right[\)\]\}]", ")", s)
    s = re.sub(r"\\cdot|\\times", " × ", s)
    s = re.sub(r"\\pm", " ± ", s)
    s = re.sub(r"\\leq", " ≤ ", s)
    s = re.sub(r"\\geq", " ≥ ", s)
    s = re.sub(r"\\neq", " ≠ ", s)
    s = re.sub(r"\\infty", "infinity", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    s = re.sub(r"  +", " ", s)
    return s.strip()


def _parse_answer_value(raw: str) -> str:
    ans = raw.strip()
    ans = re.split(
        r",?\s*(?:sorry|actually|wait|i mean|correction:?)\b",
        ans,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    return ans.rstrip(".,;")


def _extract_final_plain_answer(text: str) -> tuple[str | None, list[int]]:
    """Return the last stated answer (by line order) and indices to strip from body."""
    lines = text.splitlines()
    found: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        for m in re.finditer(
            r"(?:answer|final answer)\s*:\s*(.+?)(?:\s*$|\s*(?:,|\.\s*(?:sorry|actually|wait)\b))",
            stripped,
            re.I,
        ):
            found.append((i, _parse_answer_value(m.group(1))))
        if not any(idx == i for idx, _ in found):
            m = re.search(r"(?:answer|final answer)\s*:\s*(.+)$", stripped, re.I)
            if m:
                found.append((i, _parse_answer_value(m.group(1))))
        m = re.search(
            r"(?:actually,?\s+)?(?:the )?(?:correct |final |actual )?answer is\s+(.+?)\s*\.?\s*$",
            stripped,
            re.I,
        )
        if m:
            found.append((i, _parse_answer_value(m.group(1))))

    if found:
        return found[-1][1], sorted({idx for idx, _ in found})

    boxed = extract_answer(text)
    return (_humanize_latex(boxed) if boxed else None), []


def _is_correction_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^(?:sorry|actually|wait|correction|i (?:was )?wrong)\b", s, re.I):
        return True
    if re.search(r"\bsorry\b.*\bnot\b", s, re.I):
        return True
    if re.search(r"\b(?:actually,?\s+)?(?:the )?(?:correct|final|actual) answer is\b", s, re.I):
        return True
    return False


def humanize_quant_response(text: str) -> str:
    """Plain-text output for phone UI; one consistent final answer at the top."""
    if not text:
        return "No textual answer was returned by the model."

    body = text.strip()
    body = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", body)
    body = _humanize_latex(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    final, answer_line_indices = _extract_final_plain_answer(body)
    answer_indices = set(answer_line_indices)

    reasoning_lines: list[str] = []
    for i, line in enumerate(body.splitlines()):
        if i in answer_indices:
            continue
        if _is_correction_line(line):
            continue
        reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()

    if final:
        final = _humanize_latex(final)
        return f"Answer: {final}\n\n{reasoning}" if reasoning else f"Answer: {final}"

    return reasoning or body
