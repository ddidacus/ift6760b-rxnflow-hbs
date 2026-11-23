"""Additions to the vendored RxnFlow: a roshambo2 reward and an HBS action policy.

Nothing under ``src/rxnflow`` is modified. The HBS arm is reached by subclassing
``RxnFlowPolicy`` and ``RxnFlowTrainer``, so the baseline arm runs upstream code
exactly as cloned.
"""
