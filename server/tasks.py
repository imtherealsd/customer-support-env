import copy
from typing import List, Dict, Any

POLICY_TEXT: str = """
CUSTOMER SUPPORT POLICY:
1. If the item is defective and within 30 days of purchase, ACTION: refund.
2. If the item is defective and older than 30 days, ACTION: replace.
3. If the customer indicates they are a VIP, ACTION: escalate (overrides rules 1 and 2).
4. If a ticket requests a refund/replace but does NOT provide an order ID, ACTION: ask_info (overrides rules 1 and 2).
5. All resolved tickets must match exactly these statuses. Once all tickets are in their correct final statuses ('refunded', 'replaced', 'escalated', 'waiting_info'), you must use ACTION: done to submit.
"""

class SupportTask:
    def __init__(self, task_id: str, difficulty: str, policy: str, tickets: List[Dict[str, Any]], expected_statuses: Dict[str, str], max_steps: int):
        self.task_id = task_id
        self.difficulty = difficulty
        self.policy = policy
        self.initial_tickets = tickets
        self.expected_statuses = expected_statuses 
        self.max_steps = max_steps

    def grade(self, current_tickets: List[Dict[str, Any]]) -> float:
        """
        Returns a score between 0.01 and 0.99 based on how many tickets match their expected status.
        """
        if not self.expected_statuses:
            score = 1.0
        else:
            correct = 0
            total = len(self.expected_statuses)
            for ticket in current_tickets:
                tid = ticket["ticket_id"]
                if tid in self.expected_statuses and ticket["status"] == self.expected_statuses[tid]:
                    correct += 1
            score = correct / total
            
        return max(0.01, min(0.99, score))

easy_task = SupportTask(
    task_id="easy_refund",
    difficulty="easy",
    policy=POLICY_TEXT,
    tickets=[{
        "ticket_id": "T001",
        "customer_name": "Alice",
        "is_vip": False,
        "days_since_purchase": 10,
        "has_order_id": True,
        "issue_description": "My item is defective. I want my money back.",
        "status": "open"
    }],
    expected_statuses={"T001": "refunded"},
    max_steps=5
)

medium_task = SupportTask(
    task_id="medium_triage",
    difficulty="medium",
    policy=POLICY_TEXT,
    tickets=[
        {
            "ticket_id": "T002",
            "customer_name": "Bob",
            "is_vip": False,
            "days_since_purchase": 45,
            "has_order_id": True,
            "issue_description": "My item is defective after 6 weeks.",
            "status": "open"
        },
        {
            "ticket_id": "T003",
            "customer_name": "Charlie",
            "is_vip": False,
            "days_since_purchase": 15,
            "has_order_id": False,
            "issue_description": "My item is defective. Refund please.",
            "status": "open"
        }
    ],
    expected_statuses={"T002": "replaced", "T003": "waiting_info"},
    max_steps=10
)

hard_task = SupportTask(
    task_id="hard_mixed",
    difficulty="hard",
    policy=POLICY_TEXT,
    tickets=[
        {
            "ticket_id": "T004",
            "customer_name": "Dave",
            "is_vip": True,
            "days_since_purchase": 10,
            "has_order_id": True,
            "issue_description": "I am furious, the item is defective.",
            "status": "open"
        },
        {
            "ticket_id": "T005",
            "customer_name": "Eve",
            "is_vip": False,
            "days_since_purchase": 100,
            "has_order_id": True,
            "issue_description": "Defective item.",
            "status": "open"
        },
        {
            "ticket_id": "T006",
            "customer_name": "Frank",
            "is_vip": False,
            "days_since_purchase": 5,
            "has_order_id": False,
            "issue_description": "Where is my refund for my defective item?",
            "status": "open"
        }
    ],
    expected_statuses={"T004": "escalated", "T005": "replaced", "T006": "waiting_info"},
    max_steps=15
)

TASKS = {
    # Primary keys — match the task_id attribute and what agents should pass
    "easy_refund": easy_task,
    "medium_triage": medium_task,
    "hard_mixed": hard_task,
    # Short aliases for convenience
    "easy": easy_task,
    "medium": medium_task,
    "hard": hard_task,
}
