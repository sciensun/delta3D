"""Interfaces for Stage 1.5 identity-control and repeat reliability tests."""


def identity_control_spec(source_root, target_root, camera_names):
    return {"source_root": source_root, "target_root": target_root,
            "camera_names": list(camera_names), "purpose": "identity_control"}


def intensity_path_spec(style_family, intensities):
    return {"style_family": style_family, "intensities": [float(x) for x in intensities],
            "purpose": "intensity_path_consistency"}
