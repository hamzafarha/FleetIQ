"""Training entrypoints for Phase 6 (gradient boosting) and Phase 7
(hybrid GRU/LSTM + MLP). Left as stubs — do not implement until the
modeling approach for each phase is confirmed in docs/project_charter.md.
"""


def train_gradient_boosting(X_train, y_train, X_val, y_val, config: dict):
    """Train an XGBoost or LightGBM baseline (config picks which).

    TODO (Phase 6): implement once feature set is finalized from EDA.
    """
    raise NotImplementedError("Phase 6 — not started.")


def train_hybrid_dl(sequence_data, static_data, y_train, config: dict):
    """Train the GRU/LSTM (sequence) + MLP (static) hybrid model.

    TODO (Phase 7): blocked on the GPS-trace availability question in
    docs/project_charter.md — this branch only makes sense if raw GPS
    traces exist (real data) or the Porto Taxi fallback is used for
    prototyping.
    """
    raise NotImplementedError("Phase 7 — not started, blocked on GPS trace data availability.")
