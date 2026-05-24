"""Runtime workload-tier classification.

NOT a provider router — provider selection is external to LATTICE.
This module classifies workload complexity (SIMPLE/MEDIUM/COMPLEX/REASONING)
for setting optimisation budgets.
"""

from lattice.runtime.tier_classifier import Tier, TierClassifier, TierDecision

__all__ = ["Tier", "TierClassifier", "TierDecision"]
