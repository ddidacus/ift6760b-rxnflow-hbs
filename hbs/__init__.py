"""Experiment harness around the vendored RxnFlow: rewards, entry point, analysis.

The action policies these experiments compare live in the package itself, under
``rxnflow.gflownet.fwd`` (``hbs`` for hashed buckets, ``cas`` for k-means cells).
What stays here is what is specific to the experiments: the target rewards, the
training entry point (``python -m hbs.train``), and the offline analysis scripts.
"""
