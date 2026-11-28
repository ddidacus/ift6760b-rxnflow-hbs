"""The reference ligands under ``support/queries``, shared by every target reward.

Both reward functions in this package score against the same nine SDFs, so the
target table and the lookup live here rather than in either of them. That keeps
the 2D reward free of ``hbs.reward``, which imports ``roshambo2`` and therefore
needs a CUDA build just to read a file name.
"""

from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem

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


def find_query_dir() -> Path:
    """Locate support/queries without assuming where this package was installed.

    ROSHAMBO_QUERY_DIR wins when set; otherwise the nearest support/queries at or
    above this file is used, so the package works both inside this repository's
    baselines tree and copied into a standalone RxnFlow checkout that carries its
    own support/queries. Resolution is deferred to call time: an absolute
    sdf_path never needs it.
    """
    override = os.environ.get("ROSHAMBO_QUERY_DIR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "support" / "queries"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "no support/queries directory found above "
        f"{Path(__file__).resolve()}; set ROSHAMBO_QUERY_DIR or pass sdf_path"
    )


def resolve_query_path(target: str | None, sdf_path: str | None) -> Path:
    """The SDF a reward should read, from either a target key or an explicit path."""
    if sdf_path is None:
        if target not in TARGET_FILES:
            raise ValueError(
                f"target must be one of {sorted(TARGET_FILES)}, got {target!r}; "
                "pass sdf_path for a target outside this set"
            )
        sdf_path = str(find_query_dir() / TARGET_FILES[target])
    path = Path(sdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"query molecule not found: {path}")
    return path


def read_query_mol(path: Path) -> Chem.Mol:
    """First molecule of an SDF, with its 3D block intact."""
    molecule = Chem.SDMolSupplier(str(path))[0]
    if molecule is None:
        raise ValueError(f"failed to read a reference molecule from {path}")
    return molecule


def read_query_smiles(path: Path) -> str:
    """Canonical 2D SMILES of an SDF's first molecule.

    Round-tripping through SMILES is what drops the conformer: the SDF carries
    explicit hydrogens and coordinates, and a 2D reward must compare the target
    on the same footing as a generated SMILES, which has neither. Stereo
    annotations survive, since they are in the graph rather than the block.
    """
    flat = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(read_query_mol(path))))
    if flat is None:
        raise ValueError(f"failed to flatten the reference molecule from {path}")
    return Chem.MolToSmiles(flat)
