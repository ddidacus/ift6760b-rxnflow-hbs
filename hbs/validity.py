"""Independent replay of a recorded synthesis route: did the chemistry actually work?

Why this exists separately from the sampler's own flag
-----------------------------------------------------
``Trajectory.valid`` is the sampler's bookkeeping, and it carries only two reasons
(``invalid_transition`` / ``max_reactions``). It says a step failed, not *what* failed,
and it is reported by the same code that chose the action -- so it cannot catch a
disagreement between what the policy believed it was building and what RDKit actually
produces. This module re-derives validity from the record alone: it takes the
``reaction_smarts`` and reactants written into ``sample.jsonl``, runs the reaction again,
and checks the recorded product comes back.

It matters more for RxnFlow-CAS than for the baseline. Upstream only ever offers rows
from a library the state's action space allows, so chemical legality is enforced by
construction and this metric is ~100% by definition. The intended CAS design indexes
every synthon globally and does **not** enforce that -- a probe may open a cell holding
rows that cannot react at the current position -- so the illegal-action rate becomes a
real quantity that has to be reported rather than assumed. Measured on the current
catalog, a uniform pick from the whole catalog is chemically legal only 8.7% of the time
action-weighted, and 0.09% at the median step, so this is the number that decides whether
a global index is usable.

What is checked, per step
-------------------------
``FIRST_SYNTHON``   the recorded product parses, and is the chosen synthon placed.
``UNIRXN_*``        the one-template reaction applied to the state yields the product.
``BIRXN_*``         the two-template reaction applied to (state, synthon) yields it.
Across steps, each step's ``state`` must equal the previous step's ``product_smiles``,
and the last product must be the trajectory's ``final_smiles``.

Failures are classified, because the kinds mean different things: ``no_product`` is the
chemistry refusing the pair (the illegal-action case a global index would create),
``product_mismatch`` is the route bookkeeping disagreeing with RDKit, and ``chain_break``
is a recording bug rather than a chemical one.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

logger = logging.getLogger(__name__)

OK = "ok"
NO_PRODUCT = "no_product"
PRODUCT_MISMATCH = "product_mismatch"
CHAIN_BREAK = "chain_break"
UNPARSEABLE = "unparseable"
BAD_SMARTS = "bad_smarts"
FAILURE_KINDS = (OK, NO_PRODUCT, PRODUCT_MISMATCH, CHAIN_BREAK, UNPARSEABLE, BAD_SMARTS)


def _canonical(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Isotope-labelled synthons and partially built states are legal inputs here
        # but do not always sanitize; the atom graph is still what we compare on.
        pass
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _products(smarts: str, reactants: list[str]) -> set[str] | None:
    """Every distinct product of applying ``smarts`` to ``reactants``, or None."""
    try:
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
        if reaction is None:
            return None
        reaction.Initialize()
    except Exception:
        return None
    mols = []
    for smiles in reactants:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        mols.append(mol)
    if len(mols) != reaction.GetNumReactantTemplates():
        return None
    found: set[str] = set()
    try:
        for product_set in reaction.RunReactants(tuple(mols), 0):
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    found.add(Chem.MolToSmiles(product))
                except Exception:
                    continue
    except Exception:
        return None
    return found


def replay_step(step: dict[str, Any], expected_state: str | None) -> str:
    """Outcome of one recorded step: one of ``FAILURE_KINDS``."""
    state = step.get("state") or ""
    product = _canonical(step.get("product_smiles"))
    if product is None:
        return UNPARSEABLE
    if expected_state is not None and _canonical(state) != expected_state:
        return CHAIN_BREAK

    smarts = step.get("reaction_smarts")
    synthon = step.get("synthon_smiles")
    if not smarts:
        # First synthon: nothing is applied, the block is simply placed.
        return OK if product else UNPARSEABLE

    reactants = [state] if synthon is None else [state, synthon]
    if not state:
        return CHAIN_BREAK
    outs = _products(smarts, reactants)
    if outs is None:
        return BAD_SMARTS
    if not outs:
        return NO_PRODUCT
    return OK if product in outs else PRODUCT_MISMATCH


def replay_route(record: dict[str, Any]) -> tuple[bool, str, int]:
    """(route is valid, first failure kind, index of the failing step)."""
    steps = record.get("traj") or []
    if not steps:
        return False, CHAIN_BREAK, 0
    expected: str | None = None
    for index, step in enumerate(steps):
        outcome = replay_step(step, expected)
        if outcome != OK:
            return False, outcome, index
        expected = _canonical(step.get("product_smiles"))
    final = _canonical(record.get("final_smiles"))
    if final is not None and expected is not None and final != expected:
        return False, PRODUCT_MISMATCH, len(steps) - 1
    return True, OK, len(steps)


def validity_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pathway-validity summary over a batch of sampled trajectories."""
    if not records:
        return {"pathway_n": 0}
    kinds: Counter[str] = Counter()
    steps_ok = steps_total = routes_ok = 0
    depth: Counter[int] = Counter()
    for record in records:
        ok, kind, index = replay_route(record)
        routes_ok += int(ok)
        kinds[kind] += 1
        n = len(record.get("traj") or [])
        steps_total += n
        steps_ok += n if ok else index
        if not ok:
            depth[index] += 1
    out: dict[str, Any] = {
        "pathway_n": len(records),
        "pathway_validity": routes_ok / len(records),
        "step_validity": steps_ok / steps_total if steps_total else 0.0,
    }
    for kind in FAILURE_KINDS:
        if kind != OK and kinds.get(kind):
            out[f"pathway_{kind}"] = kinds[kind] / len(records)
    if depth:
        out["pathway_first_failure_step"] = sum(
            k * v for k, v in depth.items()
        ) / sum(depth.values())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=2000, help="routes per run")
    parser.add_argument("--quiet-rdkit", action="store_true", default=True)
    args = parser.parse_args(argv)
    if args.quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")

    header = f"{'run':40s}{'routes':>8s}{'valid':>9s}{'step':>9s}{'failures'}"
    print(header); print("-" * (len(header) + 20))
    for run in args.run_dirs:
        records = []
        for line in (run / "sample.jsonl").open():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("valid"):
                records.append(record)
            if len(records) >= args.limit:
                break
        m = validity_metrics(records)
        if not m.get("pathway_n"):
            print(f"{run.name:40s}  (no records)"); continue
        fails = ", ".join(
            f"{k.replace('pathway_', '')} {v:.1%}"
            for k, v in m.items()
            if k.startswith("pathway_") and k.split("pathway_")[1] in FAILURE_KINDS
        ) or "none"
        print(f"{run.name:40s}{m['pathway_n']:8d}{m['pathway_validity']:9.2%}"
              f"{m['step_validity']:9.2%}  {fails}")


if __name__ == "__main__":
    main()
