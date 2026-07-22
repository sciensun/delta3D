"""Stage 1 configuration and observation-mode validation."""

OBSERVATION_MODES = ("image_only", "observed_2d", "oracle_3d", "hybrid")


def validate_observation_mode(mode):
    if mode not in OBSERVATION_MODES:
        raise ValueError("observation_mode must be one of {}".format(", ".join(OBSERVATION_MODES)))
    return mode
