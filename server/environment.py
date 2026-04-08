import uuid
import copy
from typing import Optional, Any
from openenv.core.env_server import Environment

from models import SupportAction, SupportObservation, SupportState
from server.tasks import TASKS


class SupportEnvironment(Environment):
    """
    Customer Support triage environment.

    An AI agent reads a company policy and a list of open tickets, 
    then resolves each ticket by issuing the correct action (refund, replace,
    escalate, ask_info) according to business rules. Episode ends when the agent
    calls action_type='done' or max_steps is exceeded.
    """

    def __init__(self):
        super().__init__()
        self._task = TASKS["easy_refund"]
        self._current_tickets = []
        self._episode_id = ""
        self._step_count = 0
        self._done = True
        self._score = 0.01

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """Reset the environment to start a new episode.
        
        Args:
            seed: Optional random seed (not used but accepted for API compliance).
            episode_id: Optional episode identifier. Auto-generated if not provided.
            task_id: One of 'easy_refund', 'medium_triage', 'hard_mixed'.
                     Defaults to 'easy_refund'.
        """
        self._reset_rubric()

        if task_id and task_id in TASKS:
            self._task = TASKS[task_id]
        else:
            self._task = TASKS["easy_refund"]

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._score = 0.01
        self._current_tickets = copy.deepcopy(self._task.initial_tickets)

        return SupportObservation(
            company_policy=self._task.policy,
            active_tickets=self._current_tickets,
            last_action_result=None,
            last_error=None,
            done=False,
            reward=self._score,
        )

    def step(self, action: SupportAction, **kwargs: Any) -> SupportObservation:
        """Process an action and return the new observation.
        
        Reward is computed as the fraction of tickets correctly resolved.
        Scores are clamped to [0.01, 0.99].
        """
        self._step_count += 1

        last_error = None
        last_action_result = None

        if action.action_type == "done":
            self._done = True
            last_action_result = "Agent marked task as done."
        else:
            if not action.ticket_id:
                last_error = "ticket_id is required for actions other than 'done'"
            else:
                ticket_idx = next(
                    (i for i, t in enumerate(self._current_tickets)
                     if t["ticket_id"] == action.ticket_id),
                    -1
                )

                if ticket_idx == -1:
                    last_error = f"Ticket {action.ticket_id} not found."
                else:
                    action_to_status = {
                        "refund": "refunded",
                        "replace": "replaced",
                        "escalate": "escalated",
                        "ask_info": "waiting_info",
                    }
                    new_status = action_to_status.get(action.action_type, "")
                    if new_status:
                        self._current_tickets[ticket_idx]["status"] = new_status
                        last_action_result = (
                            f"Successfully set ticket {action.ticket_id} "
                            f"status to '{new_status}'."
                        )

        # Compute reward via grader
        raw_score = self._task.grade(self._current_tickets)
        self._score = raw_score  # grader already clamps to [0.01, 0.99]

        # Cap steps
        if self._step_count >= self._task.max_steps:
            self._done = True
            if not last_error:
                last_error = "Exceeded max steps."

        return SupportObservation(
            company_policy=self._task.policy,
            active_tickets=self._current_tickets,
            last_action_result=last_action_result,
            last_error=last_error,
            done=self._done,
            reward=self._score,
        )

    @property
    def state(self) -> SupportState:
        """Return current episode state metadata."""
        return SupportState(
            task_id=self._task.task_id,
            episode_id=self._episode_id,
            step_count=self._step_count,
            score=self._score,
            is_done=self._done,
        )
