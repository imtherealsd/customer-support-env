from openenv.core.env_server import create_fastapi_app

from models import SupportAction, SupportObservation
from server.environment import SupportEnvironment

# Pass the *class* as a factory — the framework will instantiate a new
# SupportEnvironment per session.  Do NOT pass an instance here.
app = create_fastapi_app(SupportEnvironment, SupportAction, SupportObservation)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
