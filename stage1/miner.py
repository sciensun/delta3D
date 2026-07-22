"""Small interfaces shared by future Stage 1 runners."""
from .config import validate_observation_mode


def validate_observation_for_mode(observation, mode):
    validate_observation_mode(mode)
    if mode == "image_only":
        return True
    observation.validate(mode=mode)
    return True
