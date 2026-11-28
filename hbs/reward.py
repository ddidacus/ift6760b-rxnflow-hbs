"""Roshambo2 shape-and-colour similarity as a RxnFlow reward.

Scores a generated molecule by the best 3D overlay it achieves against a
reference ligand -- the objective NuevoBenchmark uses. The settings are the ones
the published REINVENT, Augmented Hill Climbing and NuevaMol runs were trained on
(``num_isomers=2``, ``num_confs=25``, colour on, combination score, CUDA
backend), so the numbers this produces sit on the same scale as those, and as the
GFlowNet baseline's proxy in ``baselines/gflownet/gflownet/proxy/roshambo2.py``.

The default query is CDK8. Query SDFs are shared, at ``support/queries`` in the
top level of this package, and resolved by ``hbs.queries``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from roshambo2 import Roshambo2

from rxnflow import RewardFunction

from hbs.queries import read_query_mol, resolve_query_path

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# Conformer preparation runs in a worker pool, and RDKit drops molecule properties
# when pickling unless told otherwise. Without this the "_Name" that carries each
# molecule's batch index is lost in transit and every score comes back as zero.
Chem.SetDefaultPickleProperties(
    Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.ComputedProps
)

# ======== Conformer preparation
# Module level so the worker pool can pickle them.


def _enumerate_stereoisomers(rdmol, num_isomers: int, random_seed: int):
    options = StereoEnumerationOptions(
        maxIsomers=num_isomers, rand=random_seed, unique=True, onlyUnassigned=True
    )
    return list(EnumerateStereoisomers(rdmol, options=options))


def _embed_and_optimize(rdmol, random_seed: int):
    rdmol = Chem.AddHs(rdmol)
    if AllChem.EmbedMolecule(rdmol, maxAttempts=1000, randomSeed=random_seed) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def _generate_confs(rdmol, num_confs: int, random_seed: int):
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.clearConfs = True
    rdDistGeom.EmbedMultipleConfs(rdmol, numConfs=num_confs, params=params)
    return rdmol


def _prep_rdmol(rdmol, num_isomers: int, num_confs: int, random_seed: int):
    if rdmol is None:
        return []
    name = rdmol.GetProp("_Name") if rdmol.HasProp("_Name") else "mol"
    try:
        conformers = []
        for index, isomer in enumerate(
            _enumerate_stereoisomers(rdmol, num_isomers, random_seed)
        ):
            embedded = _embed_and_optimize(isomer, random_seed)
            if embedded is None:
                continue
            embedded.SetProp("_Name", f"{name}_{index}")
            conformers.append(_generate_confs(embedded, num_confs, random_seed))
        return conformers
    except Exception as error:  # noqa: BLE001 - one bad molecule must not stop a batch
        print(f"[!] Failed to prepare molecule {name}: {error}. Skipping.")
        return []


# ======== Reward


class Roshambo2Reward(RewardFunction):
    objectives = ("roshambo2",)

    def __init__(
        self,
        target: str | None = "cdk8",
        sdf_path: str | None = None,
        n_jobs: int | None = None,
        num_confs: int = 25,
        num_isomers: int = 2,
        random_seed: int = 109838974,
        prep_timeout: float = 300.0,
        backend: str = "cuda",
    ):
        """
        Arguments
        ---------
        target
            A key of ``hbs.queries.TARGET_FILES``, resolved against
            ``support/queries``.
            Ignored when ``sdf_path`` is given.
        sdf_path
            Any other query molecule, for a target outside that set.
        n_jobs
            Conformer-preparation processes. Preparation is this reward's dominant
            cost and it is CPU bound, so this sets the wall time and should match
            the cores the job was actually allocated: a pool wider than the
            allocation only oversubscribes. Defaults to ROSHAMBO_NUM_CPUS, or 8.
        prep_timeout
            Whole-batch ceiling on conformer preparation. RDKit's
            EmbedMultipleConfs can hang indefinitely on a pathological molecule,
            and a blocking pool call then never returns. Healthy preparation of a
            64-molecule batch takes ~0.1-2s, so this only ever cuts a true hang;
            the molecules it drops score 0.
        """
        query_path = resolve_query_path(target, sdf_path)

        self.target = target
        self.sdf_path = str(query_path)
        self.n_jobs = int(
            n_jobs if n_jobs is not None else os.environ.get("ROSHAMBO_NUM_CPUS", 8)
        )
        self.num_confs = num_confs
        self.num_isomers = num_isomers
        self.random_seed = random_seed
        self.prep_timeout = prep_timeout
        self.backend = backend

        self.query_mol = read_query_mol(query_path)

        # One pool for the run. Rebuilding it per call dominated the step time when
        # this reward was profiled in the parent package: fork cost grows with the
        # parent's address space, so a long run's steps got slower and slower.
        self._pool = multiprocessing.Pool(processes=self.n_jobs)
        # Oracle calls, for budget-matching against the other baselines. RxnFlow
        # scores unique uncached SMILES only, so this counts distinct molecules.
        self.num_oracle_calls = 0
        logger.info(
            "Roshambo2 reward ready: target=%s query=%s n_jobs=%d num_confs=%d "
            "num_isomers=%d backend=%s",
            target,
            self.sdf_path,
            self.n_jobs,
            num_confs,
            num_isomers,
            backend,
        )

    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Return the Tanimoto combination score of each molecule against the query.

        Anything RDKit cannot parse, or whose conformers cannot be generated,
        scores 0, so an unusable generation is penalised rather than dropped.
        """
        scores = np.zeros((len(smiles_list), 1), dtype=np.float32)
        self.num_oracle_calls += len(smiles_list)

        molecules = []
        for index, smiles in enumerate(smiles_list):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None and molecule.GetNumAtoms() > 0:
                molecule.SetProp("_Name", str(index))
                molecules.append(molecule)
        if not molecules:
            return scores

        dataset = self._prepare(molecules)
        if not dataset:
            return scores

        try:
            calculator = Roshambo2(
                queries=self.query_mol,
                datasets=dataset,
                color=True,
                n_cpus_prepare=self.n_jobs,
            )
            result = calculator.compute(
                backend=self.backend,
                reduce_over_conformers=True,
                optim_mode="combination",
                write_scores=False,
            )
        except Exception as error:  # noqa: BLE001 - a failed batch scores zero
            logger.warning("Roshambo2 batch error: %s. Returning zero scores.", error)
            return scores

        # Names carry each molecule's row, so the best conformer of each molecule
        # is mapped back to it; molecules absent from the result keep their 0.
        frame = next(iter(result.values()))
        frame["ID"] = frame["name"].str.split("_").str[0].astype(int)
        frame = frame.sort_values(["ID", "tanimoto_combination"], ascending=[True, False])
        frame = frame.drop_duplicates(subset="ID", keep="first")
        merged = pd.merge(
            pd.DataFrame({"ID": range(len(smiles_list))}), frame, on="ID", how="left"
        ).fillna({"tanimoto_combination": 0.0})
        values = merged["tanimoto_combination"].to_numpy(dtype=np.float32)
        # RxnFlow requires finite, non-negative rewards; Tanimoto scores are
        # already in [0, 1], so this only guards against a surprise.
        return np.clip(values, 0.0, None).reshape(-1, 1)

    def _prepare(self, molecules: list[Chem.Mol]) -> list[Chem.Mol]:
        # Submitted per molecule, under one batch deadline, so a single hanging
        # molecule is dropped instead of blocking the batch.
        arguments = [
            (molecule, self.num_isomers, self.num_confs, self.random_seed)
            for molecule in molecules
        ]
        pending = [self._pool.apply_async(_prep_rdmol, a) for a in arguments]
        deadline = time.monotonic() + self.prep_timeout
        prepared: list[Chem.Mol] = []
        num_timeout = 0
        for task in pending:
            try:
                prepared.extend(task.get(timeout=max(0.0, deadline - time.monotonic())))
            except multiprocessing.TimeoutError:
                num_timeout += 1
            except Exception as error:  # noqa: BLE001
                logger.warning("Conformer prep failed for one molecule: %s", error)
        if num_timeout:
            # A worker stuck inside EmbedMultipleConfs never returns, so the pool
            # has to be replaced or every later batch loses a worker until none
            # are left.
            logger.warning(
                "Conformer prep exceeded %.0fs for %d of %d molecules; dropping them "
                "and rebuilding the worker pool.",
                self.prep_timeout,
                num_timeout,
                len(arguments),
            )
            self._pool.terminate()
            self._pool = multiprocessing.Pool(processes=self.n_jobs)
        return prepared

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.terminate()
