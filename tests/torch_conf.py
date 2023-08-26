import torch
from pytorch_lightning import seed_everything

_MARK_REQUIRE_GPU = dict(
    condition=not torch.cuda.is_available(), reason="test requires GPU machine"
)


def reset_seed():
    seed_everything()
