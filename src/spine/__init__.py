"""SPINE — evaluation, temporal splitting and decision logic.

A shared library for the MAIB Term 4 projects (ADIL, ATHAR, QARAR, MIZAN).

SPINE contains no models. It contains the parts that are easy to get subtly wrong
and expensive to get wrong three times: the MASE denominator, the split boundary,
the critical fractile.

The six modules, and who leans on them:

:mod:`spine.splitting`
    Rolling-origin and group-aware temporal splits. Nothing here shuffles.

:mod:`spine.metrics`
    MASE, pinball loss, PR-AUC, Brier, calibration error, Qini.

:mod:`spine.decisions`
    Critical fractile, newsvendor, safety stock, cost-sensitive threshold —
    the bridge from a forecast to an action.

:mod:`spine.fairness`
    Group rates, disparity ratios, calibration by group.

:mod:`spine.cards`
    Model card generator. YAML in, markdown out, no regulatory vocabulary.

:mod:`spine.io`
    Shared data root, schema declaration, validating reader.
"""

from spine import cards, decisions, fairness, io, metrics, splitting

__version__ = "0.1.0"

__all__ = ["cards", "decisions", "fairness", "io", "metrics", "splitting"]
