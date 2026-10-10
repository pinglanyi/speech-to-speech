"""Load bundled Silero weights for the Ubuntu launcher, or the legacy Hub path."""
import os


def load_silero_model():
    if os.environ.get("S2S_PACKAGED_SILERO") == "1":
        from silero_vad import load_silero_vad

        return load_silero_vad()

    import torch

    model, _ = torch.hub.load(
        "snakers4/silero-vad:master", "silero_vad", trust_repo=True, skip_validation=True
    )
    return model
