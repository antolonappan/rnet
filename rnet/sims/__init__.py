from .simulator import RGridSimulator
from .sky import Sky
from .foreground_suite import ForegroundAugmentedSimulator
from .suite import build_training_manifest, training_quotas
__all__ = ["RGridSimulator", "Sky", "ForegroundAugmentedSimulator",
           "UnifiedSATSimulator", "build_training_manifest", "training_quotas"]


def __getattr__(name):
    if name == "UnifiedSATSimulator":
        from .unified import UnifiedSATSimulator
        return UnifiedSATSimulator
    raise AttributeError(name)
