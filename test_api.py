import httpx

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("--- Testing Reset Endpoints ---")
    res = httpx.post(f"{BASE_URL}/reset", json={"task_id": "easy_refund"})
    print("Reset response:", res.status_code)
    obs = res.json()
    print("Initial items:", obs.get("active_tickets"))
    
    print("\n--- Testing Step Action ---")
    action_payload = {
        "action_type": "refund",
        "ticket_id": "T001",
        "reason": "Because it is defective"
    }
    step_res = httpx.post(f"{BASE_URL}/step", json=action_payload)
    print("Step response code:", step_res.status_code)
    
    result = step_res.json()
    print("New observation status:", result.get("active_tickets", [])[0].get("status") if result.get("active_tickets") else None)
    print("Reward returned by step (should be 0.99):", result.get("reward"))
    print("Done:", result.get("done"))
    
    print("\n--- Testing Done Action ---")
    step_res_done = httpx.post(f"{BASE_URL}/step", json={"action_type": "done", "ticket_id": ""})
    print("Step response code:", step_res_done.status_code)
    result_done = step_res_done.json()
    print("Done:", result_done.get("done"))

if __name__ == "__main__":
    test_api()
