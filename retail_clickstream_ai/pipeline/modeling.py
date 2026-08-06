"""Model training, evaluation, and persistence — implemented in Stage 4.

Compares a current-category transition baseline, multinomial logistic
regression, and a random forest with fixed seeds. Selects the winner on
validation macro F1, evaluates it once on the held-out test month, and persists
the full preprocessing+estimator pipeline as ``model.joblib`` with metadata.
"""

from __future__ import annotations
