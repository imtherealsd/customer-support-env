"""
Baseline inference script for the Customer Support OpenEnv environment.

Uses the OpenAI client (compatible with any OpenAI-API-compatible endpoint)
to run an LLM agent against all 3 tasks and produces reproducible scores.

Required environment variables:
  API_BASE_URL   - The API endpoint for the LLM (e.g. https://api.openai.com/v1)
  MODEL_NAME     - The model identifier to use for inference
  HF_TOKEN       - Your Hugging Face / API key (used as the OpenAI API key)

Optional:
  ENV_BASE_URL   - The base URL of the running OpenEnv server (default: http://localhost:7860)

Structured stdout log format (required by the grader):
  [START] {"task_id": "...", "task_index": N}
  [STEP]  {"task_id": "...", "step": N, "action": {...}, "reward": X.XX, "done": false}
  [END]   {"task_id": "...", "score": X.XX}
"""

import json
import os
import sys
import httpx
from openai import OpenAI

# ── Configuration from environment ────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["easy_refund", "medium_triage", "hard_mixed"]


def get_action_for_observation(obs: dict) -> dict:
    """Use the LLM to decide the next action given the current observation."""
    active_tickets = obs.get("active_tickets", [])
    if not active_tickets:
        return {"action_type": "done", "ticket_id": None, "reason": "No tickets"}

    open_tickets = [t for t in active_tickets if t.get("status") == "open"]
    if not open_tickets:
        return {"action_type": "done", "ticket_id": None, "reason": "All tickets resolved"}

    policy = obs.get("company_policy", "")
    last_result = obs.get("last_action_result", "")
    last_error = obs.get("last_error", "")

    system_prompt = f"""You are a Level-1 Customer Support AI agent.
Your job is to read the company policy and resolve ONE open ticket at a time.

COMPANY POLICY:
{policy}

INSTRUCTIONS:
- Pick ONE ticket that has status "open".
- Choose the correct action according to the policy.
- Output a single JSON action. Valid action_type values: refund, replace, escalate, ask_info, done.
- Call 'done' only when ALL tickets are in their final state (not 'open').
"""

    context_parts = [f"Open tickets:\n{json.dumps(open_tickets, indent=2)}"]
    if last_result:
        context_parts.append(f"Last result: {last_result}")
    if last_error:
        context_parts.append(f"Last error: {last_error}")

    user_prompt = "\n".join(context_parts)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Take action on a single support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": ["refund", "replace", "escalate", "ask_info", "done"],
                            },
                            "ticket_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["action_type"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "take_action"}},
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        args.setdefault("reason", "")
        args.setdefault("ticket_id", None)
        return args
    except Exception as e:
        print(f"[WARN] Failed to parse LLM output: {e}", file=sys.stderr)
        return {"action_type": "done", "ticket_id": None, "reason": "parse_error"}


def run_task(task_id: str, task_index: int) -> float:
    """Run one full episode of the given task. Returns final score."""
    print("[START] " + json.dumps({"task_id": task_id, "task_index": task_index}))
    sys.stdout.flush()

    # Reset the environment
    try:
        res = httpx.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30.0,
        )
        res.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to reset env for task '{task_id}': {e}", file=sys.stderr)
        print("[END] " + json.dumps({"task_id": task_id, "score": 0.0}))
        return 0.0

    obs = res.json().get("observation", res.json())
    done = res.json().get("done", False)

    step = 0
    max_steps = 30  # safety cap

    while not done and step < max_steps:
        step += 1

        action_dict = get_action_for_observation(obs)

        try:
            step_res = httpx.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action_dict},
                timeout=60.0,
            )
            step_res.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Step {step} failed for task '{task_id}': {e}", file=sys.stderr)
            break

        result = step_res.json()
        obs = result.get("observation", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)

        print(
            "[STEP] " + json.dumps(
                {
                    "task_id": task_id,
                    "step": step,
                    "action": action_dict,
                    "reward": reward,
                    "done": done,
                }
            )
        )
        sys.stdout.flush()

    # Get final state
    try:
        state_res = httpx.get(f"{ENV_BASE_URL}/state", timeout=10.0)
        final_score = state_res.json().get("score", reward if step > 0 else 0.0)
    except Exception:
        final_score = reward if step > 0 else 0.0

    print("[END] " + json.dumps({"task_id": task_id, "score": final_score}))
    sys.stdout.flush()
    return final_score


def main():
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN / OPENAI_API_KEY is not set. LLM calls will fail.", file=sys.stderr)

    # Verify the server is reachable
    try:
        health = httpx.get(f"{ENV_BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        print(
            f"[ERROR] Cannot reach OpenEnv server at {ENV_BASE_URL}. "
            f"Start it with: uvicorn server.app:app --port 7860\nError: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    scores = {}
    for idx, task_id in enumerate(TASKS):
        score = run_task(task_id, idx)
        scores[task_id] = score

    print("\n=== Baseline Results ===", file=sys.stderr)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores)
    print(f"  Average: {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
