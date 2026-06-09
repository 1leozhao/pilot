import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from mss import mss
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from aime_preview.answer import humanize_quant_response


load_dotenv(".env.local")

app = Flask(__name__, template_folder="templates")

# In-memory conversation history per interview_id: {id: [{"user": str, "assistant": str}, ...]}
CONVERSATIONS: dict[str, list[dict[str, str]]] = {}


def default_prompt() -> str:
    return (
        "You are a DSA / low-level design helper. Read the full problem statement "
        "and any starter code visible on the screen. "
        "If there appears to be a live recording warning or message in progress on the screen, "
        "ignore it as it is simply for practice; continue as normal. "
        "CRITICAL: If there is ANY existing code, solution, or partial implementation visible on the screen, "
        "you MUST fix, modify, and extend that existing code in-place. DO NOT create a new solution from scratch. "
        "DO NOT rewrite the entire solution. Your job is to identify what's wrong or missing in the existing code "
        "and fix it, not to replace it with a completely new approach. "
        "If there is starter code, MODIFY and extend the existing code in-place rather than rewriting from scratch. "
        "Fix bugs, fill in missing pieces, and add only what is needed to pass all test cases. "
        "Preserve as much of the existing code structure and logic as possible - only change what needs to be fixed. "
        "Implement the most standard, commonly accepted and efficient "
        "Python solution (in terms of time and space complexity) for this problem, "
        "using the typical data structures and patterns seen in interview "
        "solutions. Do NOT use obscure, niche or rarely taught algorithms or data "
        "structures, even if they may be slightly more optimal in theory; prefer "
        "the approach that a strong candidate would most commonly write in an "
        "interview. "
        "If the question appears to be a JavaScript/TypeScript based frontend question, implement the solution in TypeScript. "
        "IMPORTANT: If the problem appears to be a system design or low-level design "
        "question (requiring classes, multiple methods, or object-oriented design) "
        "rather than a single function implementation, design the classes and methods "
        "with future extensibility and maintainability in mind. Consider clean APIs, "
        "separation of concerns, and reusable components that could be extended for "
        "future requirements. "
        "If there appears to be an error message or failed test case visible on the "
        "screen, or the user's code is incorrect, identify and fix the issue in the code, and add a brief inline comment "
        "right next to the line(s) where the fix was made (e.g., 'if not nums:  # Fixed: was missing edge case for empty input'). "
        "Place the comment on the same line as the fix, not as a separate comment block. "
        "Keep the code clean and idiomatic. Strip out ALL other comments from "
        "the final code (including any inline comments from the starter code), except "
        "for inline comments you add to mark error fixes. The result should contain ONLY "
        "executable Python with minimal inline comments only where errors were fixed. "
        "Do NOT wrap the code in markdown, backticks, or code fences such as ```python or ```; "
        "output must be plain Python code only. Return ONLY the final completed code that "
        "would be submitted (no explanation, minimal comments only for error fixes)."
    )


# AIME-Preview defaults: https://github.com/GAIR-NLP/AIME-Preview
QUANT_MAX_TOKENS = int(os.getenv("QUANT_MAX_TOKENS", "4000"))


def quant_vision_prompt(feedback: str = "") -> str:
    """Interview-friendly quant prompt (plain text, no LaTeX)."""
    prompt = (
        "You are a statistics, probability, brain teaser, and pattern recognition helper. "
        "Read the problem from the screenshot and reason step by step. "
        "Work through all steps and calculations FIRST. Do NOT state a final answer until "
        "you have fully finished your reasoning. "
        "After all steps are complete, end with exactly one final line: Answer: X "
        "(e.g., 'Answer: 0.25', 'Answer: 1/3', 'Answer: Image B'). "
        "State the final answer only once, at the very end. Never give an early answer at the "
        "top and never revise or contradict it afterward. "
        "Use simple notation only: write fractions as 1/2, powers as x^2, square roots as sqrt(2). "
        "Do NOT use LaTeX, \\boxed{}, dollar signs, backslashes, markdown, or code fences. "
        "Return plain text readable on a phone."
        "If there are multiple questions, answer the question that the mouse cursor is positioned next to."
        "Note that each question will only have one answer, unless otherwise specified."
        """
        Tip: For common math questions, refer to the most common formulas and properties. Most quant “Green Book” style math interviews commonly test probability (especially conditional probability and Bayes’ theorem), combinatorics (counting, permutations, combinations), expected value and variance, basic Markov chains, algebraic manipulation/logic puzzles, and occasionally number theory or optimization-style reasoning problems. Be sure to correctly answer questions in the following topics: logic, probability, combinatorics, and single variable calculus.
        Tip: In logic and reasoning puzzles, pay close attention to whether multiple sentences are meant to be interpreted as one combined statement or as separate assertions. A common mistake is to evaluate several sentences together as a single true/false claim when the puzzle expects each sentence to independently follow the speaker's truthfulness rules. If more than one answer appears to work, this interpretation is often worth checking first.
        Tip: In quantitative logic puzzles with conditions or ranks, always convert vague language like “higher rank,” “lower rank,” or “not true” into a clear ordered scale before evaluating statements, since most mistakes come from leaving comparisons informal or ambiguous.
        Tip: In multi-person logic puzzles, start by eliminating impossible identities early (like someone making a statement that cannot be true for a fixed-truth type), because reducing the state space first prevents getting trapped in misleading later casework.
        Tip: In interview-style probability or logic puzzles, always explicitly write out the assumptions you are using (for example whether statements are evaluated per sentence or as a single block), since many “multiple valid answers” only happen when hidden assumptions differ.
        Tip: In ranking or constraint puzzles, translate every statement into inequalities or formal constraints immediately, because treating relationships verbally instead of mathematically is the most common source of reasoning errors under time pressure.
        """
    )
    if feedback:
        prompt += (
            "\n\nUser feedback on the previous solution and additional requirements:\n"
            + feedback
            + "\n\nUpdate the solution accordingly while preserving the core problem statement."
        )
    return prompt


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env.local or export it in your shell."
        )
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> Anthropic:
    """Create an Anthropic client using ANTHROPIC_API_KEY."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in .env.local or export it in your shell."
        )
    return Anthropic(api_key=api_key)


def extract_openai_message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text_part = getattr(part, "text", None) or getattr(part, "content", None) or ""
            if text_part:
                parts.append(str(text_part))
        return "\n".join(parts) if parts else ""
    return ""


def format_quant_response(text: str) -> str:
    """Clean quant output for phone display."""
    cleaned = clean_model_output(text)
    return humanize_quant_response(cleaned)


def ask_quant_openai(
    image_bytes: bytes,
    history: list[dict[str, str]] | None = None,
    feedback: str = "",
    max_tokens: int | None = None,
) -> str:
    """Screenshot + AIME-Preview prompt → GPT vision (same flow as coding mode)."""
    prompt = quant_vision_prompt(feedback)
    text = ask_openai_about_image(
        image_bytes, prompt, history, max_tokens=max_tokens or QUANT_MAX_TOKENS
    )
    return format_quant_response(text)


def ask_quant_anthropic(
    image_bytes: bytes,
    history: list[dict[str, str]] | None = None,
    feedback: str = "",
    max_tokens: int | None = None,
) -> str:
    """Screenshot + AIME-Preview prompt → Claude vision (same flow as coding mode)."""
    prompt = quant_vision_prompt(feedback)
    text = ask_anthropic_about_image(
        image_bytes, prompt, history, max_tokens=max_tokens or QUANT_MAX_TOKENS
    )
    return format_quant_response(text)


def capture_screenshot_bytes() -> bytes:
    """Capture the primary screen and return PNG bytes."""
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def ask_openai_about_image(
    image_bytes: bytes,
    prompt: str,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 500,
) -> str:
    """Send screenshot + prompt to the OpenAI vision model and return text."""
    client = get_openai_client()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    messages: list[dict] = []

    if history:
        for turn in history:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                    + "\n\nYou are seeing a screenshot of my Mac screen. "
                    "Base your answer ONLY on this image.",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        max_completion_tokens=max_tokens,
    )

    text = extract_openai_message_text(response.choices[0].message)
    cleaned = clean_model_output(text)
    return cleaned or "No textual answer was returned by the model."


def ask_anthropic_about_image(image_bytes: bytes, prompt: str, history: list[dict[str, str]] | None = None, max_tokens: int = 500) -> str:
    """Send screenshot + prompt to Anthropic Claude Opus 4.1 and return text."""
    client = get_anthropic_client()

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages: list[dict] = []

    if history:
        for turn in history:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}],
                    }
                )
            if assistant_text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    }
                )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                    + "\n\nYou are seeing a screenshot of my Mac screen. "
                    "Base your answer ONLY on this image.",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
            ],
        }
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        messages=messages,
    )

    parts = []
    for block in response.content:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            parts.append(block.text)
    text = "\n".join(parts)
    cleaned = clean_model_output(text)
    return cleaned or "No textual answer was returned by the model."


def refine_openai_solution(prompt: str, history: list[dict[str, str]] | None = None) -> str:
    """Text-only refinement call for OpenAI based on current code + feedback."""
    client = get_openai_client()

    messages: list[dict] = []

    if history:
        for turn in history:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
    )

    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            text_part = getattr(part, "text", None) or getattr(part, "content", None) or ""
            if text_part:
                parts.append(str(text_part))
        text = "\n".join(parts) if parts else ""
    else:
        text = ""

    cleaned = clean_model_output(text)
    return cleaned or "No textual answer was returned by the model."


def refine_anthropic_solution(prompt: str, history: list[dict[str, str]] | None = None) -> str:
    """Text-only refinement call for Anthropic based on current code + feedback."""
    client = get_anthropic_client()

    messages: list[dict] = []

    if history:
        for turn in history:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": user_text}]}
                )
            if assistant_text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    }
                )

    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    )

    response = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=500,
        messages=messages,
    )

    parts = []
    for block in response.content:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            parts.append(block.text)
    text = "\n".join(parts)
    cleaned = clean_model_output(text)
    return cleaned or "No textual answer was returned by the model."


def clean_model_output(text: str) -> str:
    """Strip markdown code fences like ```python ... ``` or '''python ... '''."""
    if not text:
        return ""

    s = text.strip()

    # Handle common fenced code block patterns
    fence_starts = ("```python", "```", "'''python", "'''")
    if any(s.startswith(fs) for fs in fence_starts):
        lines = s.splitlines()
        # drop first line (fence / language hint)
        start_idx = 1
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```") or lines[i].strip().startswith("'''"):
                end_idx = i
                break
        s = "\n".join(lines[start_idx:end_idx]).strip()

    # Also strip any stray leading/trailing backticks on single-line outputs
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    if s.startswith("'''") and s.endswith("'''"):
        s = s[3:-3].strip()

    return s


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Capture screenshot and send to LLM."""
    body = request.get_json(silent=True) or {}
    mode = (body.get("mode") or "coding").lower()
    
    # Select prompt based on mode (quant uses AIME-Preview templates in aime_preview/)
    if mode != "quant":
        base_prompt = default_prompt()
        prompt = body.get("prompt", base_prompt)
    else:
        prompt = ""
    
    provider = (body.get("provider") or "openai").lower()
    feedback = (body.get("feedback") or "").strip()
    interview_id = body.get("interview_id")
    history: list[dict[str, str]] = []
    if interview_id:
        history = CONVERSATIONS.get(interview_id, [])

    try:
        img_bytes = capture_screenshot_bytes()

        effective_prompt = prompt
        if feedback and mode != "quant":
            effective_prompt = (
                prompt
                + "\n\nUser feedback on the previous solution and additional "
                "requirements:\n"
                + feedback
                + "\n\nCRITICAL: If there is existing code visible on the screen, you MUST fix and modify "
                "that existing code. DO NOT create a new solution from scratch. Update the code accordingly "
                "while preserving the core problem statement and all earlier instructions."
            )

        max_tokens = QUANT_MAX_TOKENS if mode == "quant" else 500

        if mode == "quant" and provider == "openai":
            answer = ask_quant_openai(img_bytes, history, feedback, max_tokens)
        elif mode == "quant" and provider == "anthropic":
            answer = ask_quant_anthropic(img_bytes, history, feedback, max_tokens)
        elif provider == "anthropic":
            answer = ask_anthropic_about_image(img_bytes, effective_prompt, history, max_tokens)
        else:
            answer = ask_openai_about_image(img_bytes, effective_prompt, history, max_tokens)

        if interview_id:
            turns = CONVERSATIONS.setdefault(interview_id, [])
            log_prompt = effective_prompt if mode != "quant" else f"[AIME-Preview quant] feedback={bool(feedback)}"
            turns.append({"user": log_prompt, "assistant": answer})

        return jsonify(
            {"ok": True, "answer": answer, "provider": provider, "interview_id": interview_id, "mode": mode}
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Generate an interview-ready explanation of the current solution."""
    body = request.get_json(silent=True) or {}
    provider = (body.get("provider") or "openai").lower()
    code = (body.get("code") or "").strip()
    interview_id = body.get("interview_id")

    if not code:
        return jsonify({"ok": False, "error": "No code to explain."}), 400

    history: list[dict[str, str]] = []
    if interview_id:
        history = CONVERSATIONS.get(interview_id, [])

    prompt = (
        "You are helping prepare for a coding interview. Below is a Python solution "
        "to a DSA problem. Provide a concise, one-line explanation for each of the following:\n"
        "1. Overall approach: [one line describing the algorithm/strategy]\n"
        "2. Time complexity: [one line with Big O notation and brief reason]\n"
        "3. Space complexity: [one line with Big O notation and brief reason]\n"
        "4. Key technique: [one line naming the main data structure or technique used]\n\n"
        "Format each as a single line starting with the label (e.g., 'Overall approach: binary search because...'). "
        "Keep each line brief and suitable for explaining to an interviewer. Do NOT include the code itself.\n\n"
        "Solution code:\n"
        + code
    )

    try:
        if provider == "anthropic":
            explanation = refine_anthropic_solution(prompt, history)
        else:
            explanation = refine_openai_solution(prompt, history)

        return jsonify(
            {"ok": True, "explanation": explanation, "provider": provider, "interview_id": interview_id}
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Refine the existing solution based on user feedback only (no new screenshot)."""
    body = request.get_json(silent=True) or {}
    provider = (body.get("provider") or "openai").lower()
    feedback = (body.get("feedback") or "").strip()
    code = (body.get("code") or "").strip()
    interview_id = body.get("interview_id")

    if not code:
        return jsonify({"ok": False, "error": "No existing code to refine."}), 400
    if not feedback:
        return jsonify({"ok": False, "error": "Feedback is empty."}), 400

    history: list[dict[str, str]] = []
    if interview_id:
        history = CONVERSATIONS.get(interview_id, [])

    base = default_prompt()
    prompt = (
        base
        + "\n\nHere is the current Python solution code that you previously produced:\n"
        + code
        + "\n\nUser feedback and requested changes:\n"
        + feedback
        + "\n\nCRITICAL: You MUST fix and modify the existing code above. DO NOT create a new solution from scratch. "
        "DO NOT rewrite the entire solution. Identify what needs to be changed based on the feedback and modify "
        "only those parts. Preserve as much of the existing code structure and logic as possible. "
        "Output ONLY the final Python code that should replace the previous solution (no explanation, no comments, no "
        "markdown or code fences)."
    )

    try:
        if provider == "anthropic":
            answer = refine_anthropic_solution(prompt, history)
        else:
            answer = refine_openai_solution(prompt, history)

        if interview_id:
            turns = CONVERSATIONS.setdefault(interview_id, [])
            turns.append({"user": prompt, "assistant": answer})

        return jsonify(
            {"ok": True, "answer": answer, "provider": provider, "interview_id": interview_id}
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from your phone on same network
    app.run(host="0.0.0.0", port=8000, debug=False)

