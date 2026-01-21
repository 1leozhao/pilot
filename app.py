import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from mss import mss
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic


load_dotenv(".env.local")

app = Flask(__name__, template_folder="templates")

# In-memory conversation history per interview_id: {id: [{"user": str, "assistant": str}, ...]}
CONVERSATIONS: dict[str, list[dict[str, str]]] = {}


def default_prompt() -> str:
    return (
        "You are a DSA / low-level design helper. Read the full problem statement "
        "and any starter code visible on the screen. If there is starter code, MODIFY and extend "
        "the existing code in-place rather than rewriting from scratch. Fix bugs, "
        "fill in missing pieces, and add only what is needed to pass all test "
        "cases. Implement the most standard, commonly accepted and efficient "
        "Python solution (in terms of time and space complexity) for this problem, "
        "using the typical data structures and patterns seen in interview "
        "solutions. Do NOT use obscure, niche or rarely taught algorithms or data "
        "structures, even if they may be slightly more optimal in theory; prefer "
        "the approach that a strong candidate would most commonly write in an "
        "interview. "
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


def capture_screenshot_bytes() -> bytes:
    """Capture the primary screen and return PNG bytes."""
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def ask_openai_about_image(image_bytes: bytes, prompt: str, history: list[dict[str, str]] | None = None) -> str:
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
        model="gpt-4.1",
        messages=messages,
        max_tokens=500,
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


def ask_anthropic_about_image(image_bytes: bytes, prompt: str, history: list[dict[str, str]] | None = None) -> str:
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
        model="gpt-4.1",
        messages=messages,
        max_tokens=500,
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
    prompt = body.get(
        "prompt",
        default_prompt(),
    )
    provider = (body.get("provider") or "openai").lower()
    feedback = (body.get("feedback") or "").strip()
    interview_id = body.get("interview_id")
    history: list[dict[str, str]] = []
    if interview_id:
        history = CONVERSATIONS.get(interview_id, [])

    try:
        img_bytes = capture_screenshot_bytes()

        effective_prompt = prompt
        if feedback:
            effective_prompt = (
                prompt
                + "\n\nUser feedback on the previous solution and additional "
                "requirements:\n"
                + feedback
                + "\n\nUpdate the code accordingly while preserving the core "
                "problem statement and all earlier instructions."
            )

        if provider == "anthropic":
            answer = ask_anthropic_about_image(img_bytes, effective_prompt, history)
        else:
            answer = ask_openai_about_image(img_bytes, effective_prompt, history)

        if interview_id:
            turns = CONVERSATIONS.setdefault(interview_id, [])
            turns.append({"user": effective_prompt, "assistant": answer})

        return jsonify(
            {"ok": True, "answer": answer, "provider": provider, "interview_id": interview_id}
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
        + "\n\nUpdate the code accordingly. Output ONLY the final Python code that "
        "should replace the previous solution (no explanation, no comments, no "
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

