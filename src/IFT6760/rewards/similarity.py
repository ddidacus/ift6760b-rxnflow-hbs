"""Morgan fingerprint Tanimoto reward against a reference ligand.

Target names match hbs/reward.py in RxnFlow-HBS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .reward import RewardFunction

RDLogger.DisableLog("rdApp.*")

TARGET_FILES = {
    "cdk8": "CDK8_insilico.sdf",
    "biib68": "BIIB68_bioactive.sdf",
    "bmsbtk_6o8i": "bmsBTK_6O8I_bioactive.sdf",
    "gsm_pf-06648671": "gsm_PF-06648671_bioactive.sdf",
    "gsm_ro7185876": "gsm_RO7185876_bioactive.sdf",
    "irak4_8ucc": "irak4_8UCC_bioactive.sdf",
    "nfkb_z9j": "NFKb_Z9j_bioactive.sdf",
    "nimbusbtk_8s99": "NimbusBTK_8S99_bioactive.sdf",
    "tyk2_8tb5": "tyk2_8TB5_bioactive.sdf",
}


class SimilarityReward(RewardFunction):
    """Tanimoto similarity to the first molecule in a reference SDF.

    target selects a file in query_dir; sdf_path overrides it. Use the SDF
    structure: its SMILES property may name a different molecule (as in CDK8).
    Default fingerprint settings match the diversity metrics.
    """

    objectives = ("similarity",)

    def __init__(
        self,
        target: str = "cdk8",
        sdf_path: str | Path | None = None,
        query_dir: str | Path = "data/queries",
        radius: int = 2,
        fp_size: int = 2048,
    ):
        path = Path(sdf_path) if sdf_path else Path(query_dir) / TARGET_FILES[target]
        self.reference = Chem.SDMolSupplier(str(path))[0]
        if self.reference is None:
            raise ValueError(f"no molecule in {path}")
        self.target = target
        self.radius = radius
        self.fp_size = fp_size
        self.reference_smiles = Chem.MolToSmiles(self.reference)
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
        self._reference_fp = self._generator.GetFingerprint(self.reference)

    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Similarity in [0, 1] per molecule; anything RDKit cannot parse scores 0."""
        scores = np.zeros((len(smiles_list), 1), dtype=np.float32)
        rows, fingerprints = [], []
        for row, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms():
                rows.append(row)
                fingerprints.append(self._generator.GetFingerprint(mol))
        if fingerprints:
            similarity = DataStructs.BulkTanimotoSimilarity(
                self._reference_fp, fingerprints
            )
            scores[rows, 0] = np.asarray(similarity, dtype=np.float32)
        return scores
