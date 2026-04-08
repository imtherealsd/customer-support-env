from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, ConfigDict
from openenv.core.env_server import Action, Observation, State


class SupportAction(Action):
    """Action an agent can take on a customer support ticket."""
    action_type: Literal["refund", "replace", "escalate", "ask_info", "done"]
    ticket_id: Optional[str] = None
    reason: Optional[str] = None


class TicketDetail(BaseModel):
    """Details of a support ticket."""
    model_config = ConfigDict(extra="allow")
    ticket_id: str
    customer_name: str
    is_vip: bool
    days_since_purchase: int
    has_order_id: bool
    issue_description: str
    status: str


class SupportObservation(Observation):
    """Current state of the tickets and the company policy.
    
    Note: `done` and `reward` are inherited from the base Observation class.
    """
    company_policy: str
    active_tickets: List[Dict[str, Any]]
    last_action_result: Optional[str] = None
    last_error: Optional[str] = None


class SupportState(State):
    """Metadata about the current episode.
    
    Note: `episode_id` and `step_count` are inherited from the base State class.
    """
    task_id: str
    score: float
    is_done: bool
