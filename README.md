# Customer Support OpenEnv

A real-world **OpenEnv** environment where an AI agent acts as a **Level 1 Customer Support Representative**, triaging customer tickets according to a strict company policy.

## Motivation

AI agents in production are increasingly asked to automate customer-facing support workflows — a genuinely high-stakes, high-value task. This environment models the exact decision-making process: read the SOP policy, inspect ticket attributes (purchase age, VIP status, order ID presence), and issue the correct resolution action. It provides a realistic benchmark for reasoning, rule-following, and structured decision-making.

## Environment Description

The agent receives a list of **open support tickets** and a **company policy text** as its observation. For each open ticket, it must call one of the following actions:

| Action | When to use |
|--------|------------|
| `refund` | Defective item within 30 days AND has order ID AND not VIP |
| `replace` | Defective item older than 30 days AND has order ID AND not VIP |
| `escalate` | Customer is a VIP (overrides refund/replace rules) |
| `ask_info` | Ticket is missing order ID (overrides refund/replace rules) |
| `done` | All tickets are in their final state — submit the episode |

## Action & Observation Spaces

### Observation: `SupportObservation`
| Field | Type | Description |
|-------|------|-------------|
| `company_policy` | `str` | The full SOP text describing resolution rules |
| `active_tickets` | `List[Dict]` | Current ticket list with `ticket_id`, `customer_name`, `is_vip`, `days_since_purchase`, `has_order_id`, `issue_description`, `status` |
| `last_action_result` | `Optional[str]` | Human-readable result of the last action |
| `last_error` | `Optional[str]` | Error message if the last action was invalid |
| `done` | `bool` | Whether the episode is complete |
| `reward` | `float` | Current score in [0.01, 0.99] |

### Action: `SupportAction`
| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `Literal[...]` | One of: `refund`, `replace`, `escalate`, `ask_info`, `done` |
| `ticket_id` | `Optional[str]` | ID of the ticket to act on (required unless `action_type=done`) |
| `reason` | `Optional[str]` | Optional justification |

## Tasks (3 Difficulties)

Graders score 0.01–0.99 based on the fraction of tickets correctly resolved.

| Task ID | Difficulty | Tickets | Description |
|---------|-----------|---------|-------------|
| `easy_refund` | Easy | 1 | Single defective item, within 30 days, has order ID → should be `refunded` |
| `medium_triage` | Medium | 2 | Mixed case: one replacement (>30 days), one missing order ID → `waiting_info` |
| `hard_mixed` | Hard | 3 | VIP escalation + replacement for old item + missing order ID |

## Reward Function

- Reward at each step = `correct_tickets / total_tickets` (partial credit)
- Clamped to `[0.01, 0.99]`  
- Provides dense signal throughout the episode (not just at completion)

## Setup & Usage

### Running Locally

```bash
# Install dependencies
pip install -e .

# Start the API server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run baseline agent
export OPENAI_API_KEY="sk-..."
python inference.py
```

### Environment Variables (Required for inference.py)

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model to use (e.g. `gpt-4o`) |
| `HF_TOKEN` | Your Hugging Face / API key |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"task_id": "easy_refund"}` |
| `/step` | POST | Execute action. Body: `{"action": {...}}` |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/schema` | GET | JSON schema for action/observation |

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### Hugging Face Deployment

```bash
openenv push --repo-id yourname/customer-support-env
```

## Baseline Scores (GPT-4o)

| Task | Expected Score |
|------|---------------|
| `easy_refund` | ~0.99 |
| `medium_triage` | ~0.99 |
| `hard_mixed` | ~0.66–0.99 |
