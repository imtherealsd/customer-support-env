"""
Standalone baseline agent script.

This script demonstrates the environment API by running a GPT-4o agent 
against all three tasks. It is a simpler version of inference.py meant
for quick local testing.

Usage:
  export OPENAI_API_KEY="sk-..."
  # Start server in another terminal: uvicorn server.app:app --port 7860
  python scripts/baseline_agent.py
"""

import os
import json
import sys
import httpx
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


def get_action_for_observation(obs: dict) -> dict:
    """Ask the LLM to pick an action for the current observation."""
    active_tickets = obs.get("active_tickets", [])
    if not active_tickets:
        return {"action_type": "done", "ticket_id": None, "reason": "No tickets"}

    open_tickets = [t for t in active_tickets if t.get("status") == "open"]
    if not open_tickets:
        return {"action_type": "done", "ticket_id": None, "reason": "All resolved"}

    policy = obs.get("company_policy", "")

    system_prompt = f"""You are a Customer Support AI. Follow the policy strictly.

POLICY:
{policy}

Pick ONE open ticket and choose the correct action. Only call 'done' when all tickets are resolved.
"""
    user_prompt = f"Open tickets:\n{json.dumps(open_tickets, indent=2)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Take action on a support ticket",
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
        print(f"Parse error: {e}", file=sys.stderr)
        return {"action_type": "done", "ticket_id": None, "reason": "error"}


def run_task(task_id: str):
    print(f"\n--- Running Task: {task_id} ---")

    res = httpx.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30.0)
    if res.status_code != 200:
        print(f"Failed to reset: {res.text}")
        return

    data = res.json()
    obs = data.get("observation", data)
    done = data.get("done", False)

    step_count = 0
    while not done and step_count < 25:
        step_count += 1
        action_dict = get_action_for_observation(obs)
        print(f"  Step {step_count}: {action_dict['action_type']} "
              f"ticket={action_dict.get('ticket_id', '-')}")

        step_res = httpx.post(
            f"{BASE_URL}/step", json={"action": action_dict}, timeout=60.0
        )
        if step_res.status_code != 200:
            print(f"  Step failed ({step_res.status_code}): {step_res.text}")
            break

        result = step_res.json()
        obs = result.get("observation", {})
        done = result.get("done", False)
        reward = result.get("reward", 0.0)
        print(f"  Reward: {reward:.4f} | Done: {done}")

    state_res = httpx.get(f"{BASE_URL}/state", timeout=10.0)
    final_score = state_res.json().get("score", 0.0)
    print(f"Final Score for {task_id}: {final_score:.4f}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY not set — API calls will fail.", file=sys.stderr)

    try:
        httpx.get(f"{BASE_URL}/health", timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {BASE_URL}. Start with: uvicorn server.app:app --port 7860\n{e}")
        sys.exit(1)

    for task in ["easy_refund", "medium_triage", "hard_mixed"]:
        run_task(task)
