from typing import List, Dict, Any
from server.tasks import SupportTask

# 1. Grab the policy we need
POLICY_TEXT = "Test Policy"

# 2. Re-create the Easy Task manually (to avoid OpenEnv SDK import issues in models)
easy_task = SupportTask(
    task_id="easy_mock",
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

# 3. Simulate Tickets
def test_graders():
    print("--- Testing SupportTask Grader logic ---")
    
    # Fully incorrect (Should not be 0.0, but clamped to 0.01)
    incorrect_tickets = [{"ticket_id": "T001", "status": "escalated"}]
    score_incorrect = easy_task.grade(incorrect_tickets)
    print(f"Score for Incorrect ticket: {score_incorrect} (Expected: 0.01)")
    
    # Fully correct (Should not be 1.0, but clamped to 0.99)
    correct_tickets = [{"ticket_id": "T001", "status": "refunded"}]
    score_correct = easy_task.grade(correct_tickets)
    print(f"Score for Correct ticket: {score_correct} (Expected: 0.99)")

if __name__ == "__main__":
    test_graders()
