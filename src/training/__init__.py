"""Training configuration and runtime components."""

from .config import PrivateTrainConfig, load_private_train_config, resolved_config_dict


def __getattr__(name: str):
    """Lazily expose trainer symbols to avoid data/preprocessing import cycles."""

    if name in {
        "EpochMetrics",
        "TrainingSummary",
        "create_optimizer_scheduler",
        "run_private_training",
        "train_epoch",
        "validate_epoch",
    }:
        from . import trainer

        return getattr(trainer, name)
    raise AttributeError(name)

__all__ = [
    "PrivateTrainConfig",
    "load_private_train_config",
    "resolved_config_dict",
    "EpochMetrics",
    "TrainingSummary",
    "create_optimizer_scheduler",
    "run_private_training",
    "train_epoch",
    "validate_epoch",
]
