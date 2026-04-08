import sys
from pydantic import BaseModel

class MockEnv:
    def __init__(self):
        pass

class MockBaseModel(BaseModel):
    pass

import openenv.core
class MockTypes:
    pass

sys.modules['openenv.core.models'] = __import__('types').ModuleType('models')
sys.modules['openenv.core.models'].Action = type('Action', (BaseModel,), {})
sys.modules['openenv.core.models'].Observation = type('Observation', (BaseModel,), {})
sys.modules['openenv.core.models'].State = type('State', (BaseModel,), {})
sys.modules['openenv.core.models'].StepResult = type('StepResult', (BaseModel,), {})
sys.modules['openenv.core.env_server'] = __import__('types').ModuleType('env_server')
sys.modules['openenv.core.env_server'].Environment = type('Environment', (), {})

from models import SupportAction, SupportObservation
from server.environment import SupportEnvironment

def run_test():
    env = SupportEnvironment()
    print("\n--- Task: Easy ---")
    obs = env.reset("easy")
    print("Initial Score:", env.state().score)
    print("Tickets:", [t["ticket_id"] for t in obs.active_tickets])
    action = SupportAction(action_type="refund", ticket_id="T001", reason="Testing easy")
    res = env.step(action)
    print("Step 1 (refund T001) Score:", env.state().score)
    
    # We will pretend the agent finishes
    action2 = SupportAction(action_type="done", ticket_id="", reason="done")
    res2 = env.step(action2)
    print("Final Score after done:", env.state().score)

if __name__ == "__main__":
    run_test()
