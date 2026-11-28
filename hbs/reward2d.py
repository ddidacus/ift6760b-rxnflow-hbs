"""2D fingerprint similarity to a reference ligand, as a RxnFlow reward.

A cheap stand-in for ``hbs.reward.Roshambo2Reward`` on the same nine targets and
the same query SDFs. Where that one scores the best 3D overlay a molecule can
reach -- enumerating stereoisomers, embedding 25 conformers each, then a CUDA
shape-and-colour alignment, ~40s for a batch of 64 -- this one compares graphs:
one fingerprint per molecule and a Tanimoto, microseconds per batch, no
conformers, no worker pool, no GPU.

That makes it a different objective, not an approximation of the same one: it
rewards a molecule for *being* the target rather than for occupying its volume,
so it cannot find a shape-matched scaffold hop. It exists so that action-space
experiments -- where the question is which blocks the policy reaches, not which
pharmacophore it matches -- stop paying for conformer generation.

The score is the Tanimoto coefficient over a single concatenated bit vector --
Morgan (radius 2, 2048 bits) + RDKit topological (2048 bits) + MACCS (166
defined keys) -- multiplied by a size-agreement factor. Concatenating before the
ratio, rather than averaging three per-block Tanimotos, weights each block by how
many bits it actually sets.

The size factor is not cosmetic. Measured on 9,002 real generations from this
catalog, the bare Tanimoto's mean score *rises monotonically with candidate size
up to ~43 heavy atoms for every one of the nine targets*, whether the target has
25 heavy atoms or 37: nfkb_z9j (25 atoms) scores 0.233 against molecules its own
size and 0.301 against 40-43 atom molecules. Bigger molecules contain more
substructures, so they intersect more of any target's bits faster than the union
penalises them, and the peak sits where the catalog runs out rather than where
the target is. That is a reward for growing, bounded only by the MW penalty, and
it is not specific to the dense RDKit block -- sparse Morgan (2.9% bit density)
peaks at 40 atoms for a 25-atom target too.

Multiplying by ``min/max`` of the two heavy-atom counts moves the peak onto the
target (mean offset 11.9 -> 2.4 atoms), *widens* the dynamic range that beta acts
on (top10/p50 1.55 -> 1.67) and makes small fragments less attractive, not more
(worst decoy 0.53x -> 0.21x of the median generation). ``size_agreement`` turns it
off or sharpens it; see the constructor.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from rxnflow import RewardFunction

from hbs.queries import read_query_smiles, resolve_query_path

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


# ======== Fingerprint
# Generators are stateless and thread-safe, and building one costs more than a
# fingerprint does, so they are module level and made once.

MORGAN_BITS = 2048
RDKIT_BITS = 2048
# MACCS index 0 is unused, as in src/rxnflow/envs/features.py; only the 166
# defined keys are concatenated.
MACCS_BITS = 166
FINGERPRINT_DIM = MORGAN_BITS + RDKIT_BITS + MACCS_BITS

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
_RDKIT = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=RDKIT_BITS)


def _bits(vector, size: int) -> NDArray[np.uint8]:
    array = np.zeros(size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(vector, array)
    return array


def mol_fingerprint(mol: Chem.Mol) -> NDArray[np.uint8]:
    """The concatenated Morgan + RDKit + MACCS bit vector of one molecule."""
    maccs = _bits(MACCSkeys.GenMACCSKeys(mol), MACCS_BITS + 1)
    return np.concatenate(
        [
            _bits(_MORGAN.GetFingerprint(mol), MORGAN_BITS),
            _bits(_RDKIT.GetFingerprint(mol), RDKIT_BITS),
            maccs[1:],
        ]
    )


def smiles_fingerprint(smiles: str) -> tuple[NDArray[np.uint8], int] | None:
    """Fingerprint and heavy-atom count from SMILES, or None if RDKit cannot use it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return mol_fingerprint(mol), mol.GetNumHeavyAtoms()
    except Exception as error:  # noqa: BLE001 - one bad molecule must not stop a batch
        print(f"[!] Failed to fingerprint {smiles}: {error}. Scoring 0.")
        return None


def size_agreement(
    sizes: NDArray, target_size: float, gamma: float = 1.0
) -> NDArray:
    """``(min/max)**gamma`` of two heavy-atom counts: 1 at equal size, 0 far apart.

    Symmetric, so it penalises a molecule for being larger than the target as
    much as for being smaller -- which is the point, since the bare Tanimoto
    penalises only the second. gamma sharpens it; 0 disables it.
    """
    smaller = np.minimum(sizes, target_size)
    larger = np.maximum(sizes, target_size)
    return np.divide(
        smaller, larger, out=np.zeros_like(smaller, dtype=np.float64), where=larger > 0
    ) ** gamma


def tanimoto(candidates: NDArray[np.uint8], target: NDArray[np.uint8]) -> NDArray:
    """Row-wise Tanimoto of a (n, D) bit matrix against one (D,) bit vector.

    |A & B| / |A | B| on binary input, which is in [0, 1] by construction -- no
    rescaling is applied or needed. An all-zero pair has an empty union; it is
    scored 0 rather than left as a division by zero, which is the right reading
    here since such a molecule carries no structure to match.
    """
    intersection = candidates.astype(np.float32) @ target.astype(np.float32)
    union = candidates.sum(axis=1) + target.sum() - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


# ======== Reward


class Mol2DSimilarityReward(RewardFunction):
    """2D similarity to one reference ligand.

    Named for ``2DMolSimilarityReward``, which is not a legal Python identifier.
    """

    objectives = ("sim2d",)

    def __init__(
        self,
        target: str | None = "cdk8",
        sdf_path: str | None = None,
        size_agreement: float = 1.0,
    ):
        """
        Arguments
        ---------
        target
            A key of ``hbs.queries.TARGET_FILES``, resolved against
            ``support/queries``. Ignored when ``sdf_path`` is given.
        sdf_path
            Any other query molecule, for a target outside that set.
        size_agreement
            Exponent on the heavy-atom ``min/max`` factor. 1.0 puts the score's
            peak on the target's own size, which is what the bare Tanimoto fails
            to do (see the module docstring). 0 disables the factor and recovers
            the bare Tanimoto -- useful only for reproducing a run made before
            this existed. 2.0 pins the peak harder (mean offset 1.3 atoms rather
            than 2.4) at the cost of letting size speak over chemistry.
        """
        if size_agreement < 0:
            raise ValueError(
                f"size_agreement must be non-negative, got {size_agreement}"
            )
        query_path = resolve_query_path(target, sdf_path)
        self.target = target
        self.sdf_path = str(query_path)
        # The 2D graph of the reference, taken from the same SDF the 3D reward
        # overlays against, so the two objectives share a target definition.
        self.target_smiles = read_query_smiles(query_path)
        target_mol = Chem.MolFromSmiles(self.target_smiles)
        self.target_fingerprint = mol_fingerprint(target_mol)
        self.target_size = target_mol.GetNumHeavyAtoms()
        self.size_agreement = float(size_agreement)
        # Oracle calls, for budget-matching against the other rewards. RxnFlow
        # scores unique uncached SMILES only, so this counts distinct molecules.
        self.num_oracle_calls = 0
        logger.info(
            "2D similarity reward ready: target=%s query=%s smiles=%s "
            "fingerprint=%d bits (%d set) heavy_atoms=%d size_agreement=%.2f",
            target,
            self.sdf_path,
            self.target_smiles,
            FINGERPRINT_DIM,
            int(self.target_fingerprint.sum()),
            self.target_size,
            self.size_agreement,
        )

    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Return each molecule's similarity to the target, in [0, 1].

        Tanimoto over the concatenated fingerprint, times the heavy-atom size
        agreement. Both factors are in [0, 1], so the product is too.

        Anything RDKit cannot parse scores 0, so an unusable generation is
        penalised rather than dropped.
        """
        scores = np.zeros((len(smiles_list), 1), dtype=np.float32)
        self.num_oracle_calls += len(smiles_list)

        rows, fingerprints, sizes = [], [], []
        for index, smiles in enumerate(smiles_list):
            prepared = smiles_fingerprint(smiles)
            if prepared is not None:
                fingerprint, heavy_atoms = prepared
                rows.append(index)
                fingerprints.append(fingerprint)
                sizes.append(heavy_atoms)
        if not rows:
            return scores

        values = tanimoto(np.stack(fingerprints), self.target_fingerprint)
        if self.size_agreement:
            values = values * size_agreement(
                np.asarray(sizes, dtype=np.float64),
                float(self.target_size),
                self.size_agreement,
            )
        scores[rows, 0] = values.astype(np.float32)
        return scores
