from models import SupportAction
from server.environment import SupportEnvironment
import json

def run_test():
    env = SupportEnvironment()
    
    print("\n--- Task: Easy ---")
    obs = env.reset("easy")
    print("Initial Score:", env.state().score)
    print("Tickets:", [t["ticket_id"] for t in obs.active_tickets])
    action = SupportAction(action_type="refund", ticket_id="T001", reason="Testing easy")
    res = env.step(action)
    print("Step 1 (refund T001) Score:", env.state().score)
    
    action2 = SupportAction(action_type="done")
    res2 = env.step(action2)
    print("Final Score after done:", env.state().score)

    print("\n--- Task: Medium ---")
    obs = env.reset("medium")
    print("Initial Score:", env.state().score)
    
    # Refund a wrong one to see score drops
    action = SupportAction(action_type="refund", ticket_id="T002")
    res = env.step(action)
    print("Step 1 (refund T002 wrong) Score:", env.state().score)
    
    # Fix it
    action = SupportAction(action_type="replaced", ticket_id="T002")
    # Actually action is 'replace'
    action = SupportAction(action_type="replace", ticket_id="T002")
    res = env.step(action)
    print("Step 2 (replace T002 - correct!) Score:", env.state().score)
    
if __name__ == "__main__":
    run_test()
