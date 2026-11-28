"""PMO benchmark oracles as RxnFlow rewards.

The Practical Molecular Optimization benchmark (Gao et al., NeurIPS 2022,
*Sample Efficiency Matters*) is 23 single-objective tasks drawn from GuacaMol and
from TDC's activity classifiers, all scoring in ``[0, 1]``, all evaluated under a
fixed oracle-call budget. The headline number is **AUC top-10**: the area under
the (oracle calls, mean reward of the best 10 distinct molecules) curve, divided
by the budget, which rewards finding good molecules *early* rather than at all.
``hbs/analyze.py`` already computes exactly that, so a run made here is scored
with the same code as the roshambo2 runs.

Oracles come from ``PyTDC``. Three of them (``gsk3b``, ``jnk3``, ``drd2``) are
pickled sklearn models rather than RDKit descriptors, and TDC ships pickles
written by sklearn 0.22; they need repairing before they give usable scores under
a modern stack. The repairs here are the ones in the companion package
(``biogen-mila``, ``molgen/rewards/tdcreward.py``), so a task run here produces
the same numbers as the baselines measured there:

1. ``rdkit.six`` was removed from RDKit years ago and TDC still imports it. A
   stub is installed in ``sys.modules`` before TDC's chemistry module loads,
   instead of editing the installed package.
2. The pickles predate three sklearn schema changes and will not unpickle at all
   without help: the ``missing_go_to_left`` tree-node field (1.3), the
   ``base_estimator`` -> ``estimator`` rename (1.2), and ``monotonic_cst`` on the
   decision trees (1.4). ``_CompatUnpickler`` fills them in while loading.
3. The random forests keep raw class counts in each tree's ``value``. sklearn
   < 1.4 normalised those inside ``predict_proba``; later versions assume they
   are already normalised, so the oracle returns count sums (scores in the
   thousands). Renormalising in place reproduces the pre-1.4 values exactly.
4. TDC's ``drd2`` ends in ``float(score)`` on a shape-``(1,)`` array, which
   numpy >= 2 rejects. ``Oracle.__call__`` swallows that on the single-SMILES
   path and returns 0.0 for every molecule, so the reward would silently carry
   no signal; the conversion is replaced, the model is not.

The companion package applies (2) by rewriting the installed
``tdc/chem_utils/oracle/oracle.py`` (its ``scripts/patch_tdc.py``). Here the same
fix is applied from inside this module, so nothing in ``site-packages`` is
edited and a rebuilt environment needs no extra step.

Two further constraints are worth knowing before changing the environment:

* **scikit-learn must stay at 1.7.x.** 1.8 removed the back-compatible unpickling
  paths that even ``_CompatUnpickler`` relies on, and ``gsk3b``/``jnk3`` then
  fail to load however they are read. The failure is quiet in the worst way:
  ``Oracle.__call__``'s single-SMILES path catches it and returns 0.0.
  ``_check_sklearn_version`` fails loudly rather than letting that happen.
* **The oracle pickles are looked up by a hardcoded relative path** inside TDC
  (``"oracle/gsk3b_current.pkl"``), so a run started from any other directory
  would re-download them. ``_use_oracle_dir`` redirects both the download and
  the load to an absolute directory, so this reward is independent of the
  working directory.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from packaging import version

from rxnflow import RewardFunction

logger = logging.getLogger(__name__)

# The 23 PMO tasks, in the order the paper tabulates them. Every one scores in
# [0, 1] and is larger-is-better, which is what RxnFlow requires of a reward.
PMO_TASKS: tuple[str, ...] = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)

# Oracles backed by a pickled sklearn model rather than RDKit descriptors, and so
# the ones needing the repairs above.
MODEL_BACKED_TASKS: tuple[str, ...] = ("drd2", "gsk3b", "jnk3")

# Probe molecules that score differently on all three model-backed oracles, so a
# degenerate oracle is caught at construction instead of training against a flat
# reward for hours.
PROBE_SMILES: tuple[str, ...] = (
    "CCO",  # ethanol, inactive everywhere
    "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",  # haloperidol, DRD2 active
    "O=C1NC(=O)C(c2ccc(Cl)cc2Cl)=C1c1cn(C)c2ccccc12",  # SB216763, GSK3B inhibitor
)

# Scores these probes must reproduce, to 2 decimal places, for the model-backed
# oracles. Measured in biogen-mila (PyTDC 1.1.14, scikit-learn 1.7.2, numpy 2.4.4)
# with the same repairs; they pin the repairs to known-good output rather than
# only to a plausible range. SB216763 is a published GSK3B inhibitor and
# haloperidol a published D2 antagonist, so these are chemically right as well as
# merely reproducible.
PROBE_EXPECTED: dict[str, tuple[float, ...]] = {
    "gsk3b": (0.00, 0.00, 0.99),
    "jnk3": (0.00, 0.00, 0.24),
    "drd2": (0.01, 1.00, 0.00),
}

# scikit-learn versions whose unpickling still understands TDC's 0.22-era models.
SKLEARN_MIN = version.parse("1.3")
SKLEARN_MAX_EXCLUSIVE = version.parse("1.8")


# ======== Environment repairs
# All of these run before the first Oracle is built, and all are idempotent.


def _install_rdkit_six_shim() -> None:
    """Satisfy TDC's ``from rdkit.six import iteritems``.

    ``rdkit.six`` was dropped from RDKit long ago, and
    ``tdc.chem_utils.oracle.oracle`` still imports it inside a bare ``try`` whose
    ``except`` re-raises as "Please install rdkit", so the real cause is hidden
    and *every* oracle in that module becomes unavailable. Only ``iteritems`` is
    used (one call, in a fingerprint helper).
    """
    if "rdkit.six" in sys.modules:
        return
    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda mapping: mapping.items()  # type: ignore[attr-defined]
    shim.string_types = (str,)  # type: ignore[attr-defined]
    sys.modules["rdkit.six"] = shim


def _check_sklearn_version() -> None:
    """Refuse to run against a scikit-learn that cannot read TDC's pickles.

    1.8 removed the legacy unpickling paths. The failure is not clean: the
    single-SMILES path of ``Oracle.__call__`` catches the exception and returns
    0.0, so a run would train against an all-zero reward and only the list path
    would raise. Checking the version is the only way to catch it up front.
    """
    import sklearn

    found = version.parse(sklearn.__version__)
    if not (SKLEARN_MIN <= found < SKLEARN_MAX_EXCLUSIVE):
        raise RuntimeError(
            f"scikit-learn {found} cannot load TDC's oracle pickles "
            f"(need >={SKLEARN_MIN}, <{SKLEARN_MAX_EXCLUSIVE}). 1.8 dropped the "
            "7-field tree-node dtype and the base_estimator alias these models "
            "rely on. Install scikit-learn==1.7.2; see hbs/pmo.py."
        )


def find_oracle_dir() -> Path:
    """Where TDC's oracle checkpoints live.

    ``TDC_ORACLE_DIR`` wins when set; otherwise the nearest ``oracle/`` at or
    above this file, falling back to creating one beside the package root. This
    mirrors ``hbs.reward._find_query_dir`` so the two rewards locate their data
    the same way.
    """
    override = os.environ.get("TDC_ORACLE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "oracle"
        if candidate.is_dir():
            return candidate
    # hbs/pmo.py -> repository root. First run downloads into it.
    return here.parent.parent / "oracle"


# The 8-field node struct sklearn >= 1.3 expects. TDC's pickles carry the
# 7-field one, and Tree.__setstate__ compares dtypes exactly rather than
# migrating, so the field has to be there before it is handed over.
_TREE_NODE_DTYPE = np.dtype(
    {
        "names": [
            "left_child",
            "right_child",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "missing_go_to_left",
        ],
        "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8", "u1"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 64,
    }
)


class _CompatUnpickler(pickle._Unpickler):  # type: ignore[misc]
    """Unpickler that migrates sklearn 0.22-era models forward while loading.

    Three schema gaps are filled as each object's state is applied:

    * ``Tree``: add ``missing_go_to_left`` to the node array (sklearn 1.3). The
      value chosen only decides where a NaN feature would be routed; these
      models are fed binary fingerprints and never see one, so it is immaterial
      to the scores. It is set to 1 to match the companion package exactly.
    * ``RandomForest*``: mirror ``base_estimator`` into ``estimator`` (the
      parameter was renamed in sklearn 1.2, and 1.7's ``__sklearn_tags__``
      reads the new name while fitting nothing).
    * ``DecisionTree*``: add ``monotonic_cst=None`` (sklearn 1.4).

    This has to subclass the *pure-Python* unpickler: the C implementation
    performs ``BUILD`` without any interceptable Python call, so there is no way
    to see the state before ``Tree.__setstate__`` rejects it. The cost is a
    slower load of these ~10-35MB files, once per run.
    """

    def load_build(self):  # noqa: D102 - pickle opcode handler
        if len(self.stack) >= 2:
            state = self.stack[-1]
            instance = self.stack[-2]
            if isinstance(state, dict):
                state = self._migrate(instance, state)
                self.stack[-1] = state
        super().load_build()

    @staticmethod
    def _migrate(instance: Any, state: dict) -> dict:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.tree._tree import Tree

        if isinstance(instance, Tree):
            nodes = state.get("nodes")
            if (
                isinstance(nodes, np.ndarray)
                and nodes.dtype.names is not None
                and "missing_go_to_left" not in nodes.dtype.names
            ):
                migrated = np.zeros(nodes.shape, dtype=_TREE_NODE_DTYPE)
                for field in nodes.dtype.names:
                    migrated[field] = nodes[field]
                migrated["missing_go_to_left"] = 1
                state = dict(state, nodes=migrated)
        elif isinstance(instance, (RandomForestClassifier, RandomForestRegressor)):
            if "base_estimator" in state and "estimator" not in state:
                state = dict(state, estimator=state["base_estimator"])
        elif isinstance(instance, (DecisionTreeClassifier, DecisionTreeRegressor)):
            if "monotonic_cst" not in state:
                state = dict(state, monotonic_cst=None)
        return state

    dispatch = pickle._Unpickler.dispatch.copy()  # type: ignore[attr-defined]
    dispatch[ord(b"b")] = load_build


def load_pickled_model(path: Path | str):
    """Load one of TDC's pickled oracle models, migrating it forward."""
    with open(path, "rb") as handle:
        return _CompatUnpickler(handle).load()


def _use_oracle_dir(oracle_dir: Path) -> None:
    """Make TDC read and write oracle checkpoints in ``oracle_dir``.

    TDC hardcodes ``"oracle/<name>.pkl"``, relative to the working directory, in
    both the downloader (``tdc.utils.load.oracle_load``, reached through
    ``tdc.oracles``) and the loader (``load_pickled_model``, called by every
    model-backed oracle). Patching those two module attributes -- rather than
    chdir'ing or dropping a symlink into the working directory -- makes the
    lookup absolute without touching anything on disk, and routes every model
    load through ``_CompatUnpickler``.
    """
    oracle_dir.mkdir(parents=True, exist_ok=True)

    import tdc.oracles
    import tdc.utils.load
    from tdc.chem_utils.oracle import oracle as tdc_oracle

    if getattr(tdc.oracles, "_hbs_oracle_dir", None) == str(oracle_dir):
        return

    tdc.oracles.oracle_load = partial(tdc.utils.load.oracle_load, path=str(oracle_dir))
    # Called as load_pickled_model("oracle/gsk3b_current.pkl"); only the basename
    # is meaningful, the directory is ours.
    tdc_oracle.load_pickled_model = lambda name: load_pickled_model(
        oracle_dir / Path(name).name
    )
    tdc.oracles._hbs_oracle_dir = str(oracle_dir)
    logger.info("TDC oracle checkpoints: %s", oracle_dir)


def _restore_forest_probabilities(oracle) -> int:
    """Renormalise the random-forest oracles (``gsk3b``, ``jnk3``) into [0, 1].

    Their pickles keep raw class counts in each tree's ``value``. sklearn < 1.4
    normalised those per tree inside ``predict_proba``; >= 1.4 assumes they are
    already normalised, so the oracle returns count sums instead of
    probabilities -- row sums in the thousands, scores far above 1. That makes
    the task incomparable with every other PMO task, which all score in [0, 1],
    and with the published baselines, which pin sklearn 1.2. Renormalising in
    place reproduces the pre-1.4 values exactly. Idempotent, and a no-op for
    ``drd2``, which is an SVC.

    Returns the number of trees it rescaled, for the log line.
    """
    from tdc.chem_utils.oracle import oracle as tdc_oracle

    fixed = 0
    # gsk3b/drd2 cache their model in a module global, jnk3 on the evaluator.
    for holder in (tdc_oracle, getattr(oracle, "evaluator_func", None)):
        for attribute in dir(holder):
            if not attribute.endswith("_model"):
                continue
            for tree in getattr(getattr(holder, attribute, None), "estimators_", []):
                value = tree.tree_.value
                totals = value.sum(axis=2, keepdims=True)
                if np.allclose(totals, 1.0):
                    continue
                np.divide(value, np.where(totals == 0, 1, totals), out=value)
                fixed += 1
    return fixed


def _patch_scalar_conversion(oracle, task: str) -> bool:
    """Rebind ``drd2`` to an evaluator that is safe under numpy >= 2.

    TDC's ``drd2`` ends in ``float(score)`` where ``score`` has shape ``(1,)``.
    numpy >= 2 raises ``TypeError`` for that; ``Oracle.__call__`` catches it on
    the single-SMILES path and returns ``default_property`` (0.0), so every
    molecule scores exactly 0 and the reward carries no signal, while on the list
    path the ``TypeError`` propagates and kills the run. The model and the
    fingerprint are fine, so only the conversion is replaced.
    """
    if task != "drd2":
        return False

    from rdkit import Chem
    from tdc.chem_utils.oracle import oracle as tdc_oracle
    from tdc.chem_utils.oracle.oracle import fingerprints_from_mol, load_drd2_model

    model = getattr(tdc_oracle, "drd2_model", None)
    if model is None:
        model = load_drd2_model()

    def drd2_scalar_safe(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return float(model.predict_proba(fingerprints_from_mol(mol))[0, 1])

    oracle.evaluator_func = drd2_scalar_safe
    return True


def build_oracle(task: str, oracle_dir: Path | str | None = None):
    """Return a repaired, verified TDC oracle for a PMO task.

    Verification is not optional for the model-backed oracles: both of the
    failure modes above produce a *usable-looking* oracle that scores every
    molecule 0.0, which a training run cannot distinguish from a hard task.
    """
    if task not in PMO_TASKS:
        raise ValueError(
            f"{task!r} is not a PMO task; expected one of {', '.join(PMO_TASKS)}"
        )
    _install_rdkit_six_shim()
    _check_sklearn_version()
    directory = Path(oracle_dir) if oracle_dir is not None else find_oracle_dir()
    _use_oracle_dir(directory.expanduser().resolve())

    from tdc import Oracle

    oracle = Oracle(name=task)
    if task not in MODEL_BACKED_TASKS:
        return oracle

    oracle(PROBE_SMILES[0])  # forces the pickle to load before it is patched
    rescaled = _restore_forest_probabilities(oracle)
    _patch_scalar_conversion(oracle, task)

    # Both paths are exercised: the list path is the one training uses, and the
    # single path is the one that hides exceptions behind a 0.0.
    singles = [float(oracle(smiles)) for smiles in PROBE_SMILES]
    batched = [float(value) for value in oracle(list(PROBE_SMILES))]
    if not np.allclose(singles, batched, atol=1e-9):
        raise RuntimeError(
            f"{task} oracle disagrees between its single-SMILES and list paths "
            f"({singles} vs {batched}); one of them is swallowing an exception."
        )
    if not all(0.0 <= score <= 1.0 for score in singles):
        raise RuntimeError(
            f"{task} oracle scores outside [0, 1] ({singles}), so it is not on the "
            f"PMO scale. Check scikit-learn against the pickles in {directory}."
        )
    if len(set(singles)) == 1:
        raise RuntimeError(
            f"{task} oracle returns a constant {singles[0]} for every probe "
            f"molecule, so the reward carries no signal. Check numpy and "
            f"scikit-learn against the pickles in {directory}."
        )
    expected = PROBE_EXPECTED[task]
    if not np.allclose(singles, expected, atol=0.005):
        raise RuntimeError(
            f"{task} oracle gives {singles} on the probe molecules, expected "
            f"{list(expected)}. The scores are plausible but not the published "
            "ones, so runs would not be comparable with the baselines; check "
            "PyTDC, scikit-learn and numpy versions."
        )
    logger.info(
        "%s oracle verified on %d probe molecules (%d forest trees rescaled)",
        task,
        len(PROBE_SMILES),
        rescaled,
    )
    return oracle


# ======== Reward


class OracleBudgetExhausted(RuntimeError):
    """Raised when a run has spent its PMO oracle budget.

    PMO's protocol is a fixed number of oracle calls (10,000 in the paper), not a
    fixed number of training steps, so this is a normal end to a run rather than
    an error. ``hbs/train.py`` catches it and records the run as complete.
    """

    def __init__(self, task: str, budget: int):
        super().__init__(f"{task}: oracle budget of {budget:,} calls spent")
        self.task = task
        self.budget = budget


class PMOReward(RewardFunction):
    """A single PMO task as a RxnFlow reward.

    The objective is named after the task, so ``training.jsonl`` and
    ``sample.jsonl`` carry e.g. ``objective_rewards.jnk3`` and a run is
    self-describing.

    Configure it through ``reward.settings`` in the YAML:

    ```yaml
    reward:
      beta: "32"
      settings:
        task: jnk3
        max_oracle_calls: 10000
    ```

    Note on comparability: PMO scores a molecule by the oracle alone, so these
    configs set no ``property_penalty``. With none set and a single objective,
    the ``reward`` recorded per molecule *is* the oracle score, which is what
    ``hbs/analyze.py`` integrates into AUC top-10.
    """

    def __init__(
        self,
        task: str = "jnk3",
        oracle_dir: str | None = None,
        max_oracle_calls: int | None = None,
    ):
        """
        Arguments
        ---------
        task
            One of ``PMO_TASKS``.
        oracle_dir
            Directory holding TDC's oracle checkpoints. Defaults to the nearest
            ``oracle/`` above this file, or ``TDC_ORACLE_DIR``. Missing
            checkpoints are downloaded into it on first use, so a run on a node
            without network access needs them already present -- the three
            model-backed ones are ~74MB in total and nothing else needs a
            download.
        max_oracle_calls
            PMO's budget. ``None`` means unlimited and lets the step count decide
            the run's length. When set, the reward raises
            ``OracleBudgetExhausted`` on the call that would exceed it, which
            ``hbs/train.py`` treats as a finished run.
        """
        self.task = task
        self.objectives: tuple[str, ...] = (task,)
        self.oracle_dir = str(
            Path(oracle_dir).expanduser().resolve()
            if oracle_dir is not None
            else find_oracle_dir()
        )
        if max_oracle_calls is not None and max_oracle_calls <= 0:
            raise ValueError("max_oracle_calls must be positive or None")
        self.max_oracle_calls = max_oracle_calls
        self.oracle = build_oracle(task, self.oracle_dir)
        # RxnFlow scores unique uncached SMILES only, so this counts distinct
        # molecules -- the quantity PMO budgets and hbs/analyze.py integrates over.
        self.num_oracle_calls = 0
        logger.info(
            "PMO reward ready: task=%s oracle_dir=%s budget=%s",
            task,
            self.oracle_dir,
            f"{self.max_oracle_calls:,}" if self.max_oracle_calls else "unlimited",
        )

    @property
    def calls_remaining(self) -> int | None:
        if self.max_oracle_calls is None:
            return None
        return max(0, self.max_oracle_calls - self.num_oracle_calls)

    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Score a batch of molecules on the task's oracle.

        Anything RDKit cannot parse, and anything the oracle cannot evaluate,
        scores 0 -- penalised rather than dropped, as elsewhere in this package.
        """
        if self.max_oracle_calls is not None and self.num_oracle_calls >= (
            self.max_oracle_calls
        ):
            raise OracleBudgetExhausted(self.task, self.max_oracle_calls)
        self.num_oracle_calls += len(smiles_list)

        scores = np.zeros((len(smiles_list), 1), dtype=np.float32)
        try:
            # The list path is the oracle's own batched path; it keeps order.
            values = self.oracle(list(smiles_list))
        except Exception as error:  # noqa: BLE001 - fall back to per-molecule
            logger.warning(
                "%s oracle failed on a batch of %d (%s); scoring one at a time.",
                self.task,
                len(smiles_list),
                error,
            )
            values = []
            for smiles in smiles_list:
                try:
                    values.append(float(self.oracle(smiles)))
                except Exception:  # noqa: BLE001 - one bad molecule scores 0
                    values.append(0.0)
        if np.isscalar(values):
            values = [values]
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape[0] != len(smiles_list):
            raise RuntimeError(
                f"{self.task} oracle returned {array.shape[0]} scores for "
                f"{len(smiles_list)} molecules"
            )
        # Non-finite would abort training inside RewardFunction.run; PMO scores
        # are already in [0, 1], so this only guards against a surprise.
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        scores[:, 0] = np.clip(array, 0.0, 1.0)
        return scores
