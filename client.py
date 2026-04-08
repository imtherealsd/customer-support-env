from openenv.core.env_client import EnvClient
from models import SupportAction, SupportObservation, SupportState

class SupportEnvClient(EnvClient[SupportAction, SupportObservation, SupportState]):
    """Client for interacting with the Customer Support Environment."""
    pass
