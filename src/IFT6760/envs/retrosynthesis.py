# Source: src/rxnflow/envs/retrosynthesis.py
"""Retrosynthetic route enumeration for synthon trajectories."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from IFT6760.core.molecule import Molecule
from IFT6760.core.synthon import typed_dummy_isotopes
from IFT6760.core.types import Action, ActionType, BackwardEdge, BackwardTrajectory

if TYPE_CHECKING:
    from IFT6760.core.reaction import BiReaction, UniReaction
    from IFT6760.envs.env import SynthesisEnv


_HYDROGEN_ATOM = Chem.MolFromSmarts("[#1]")


class Worker:
    """Find catalog-supported routes within reaction and synthon limits.

    Finding a route lowers the synthon bound for further search. Known routes
    above that bound are retained if they satisfy the original limits.
    Results are reverse-ordered edge lists, without probability weights.
    """

    def __init__(self, env: SynthesisEnv, prune: bool = True):
        self.prune: bool = prune
        self.max_reactions: int = env.max_reactions
        self.max_synthons: int = env.max_synthons
        self.uni_reactions: dict[str, UniReaction] = env.uni_reactions
        self.bi_reactions: dict[str, BiReaction] = env.bi_reactions
        self.brick_types = set(env.brick_types)
        self.synthon_search = {
            library_name: {smiles: index for index, smiles in enumerate(library.smiles)}
            for library_name, library in env.synthons.items()
        }
        self._uni_candidates: dict[
            tuple[int, ...], list[tuple[str, UniReaction, Action]]
        ] = {}
        for name, reaction in env.uni_reactions.items():
            signature = () if reaction.output_type is None else (reaction.output_type,)
            action = Action(
                ActionType.UNIRXN_TERMINAL
                if reaction.output_type is None
                else ActionType.UNIRXN_TRANSFORM,
                reaction=name,
            )
            self._uni_candidates.setdefault(signature, []).append(
                (name, reaction, action)
            )
        self._bi_actions = {
            subspace.name
            for space in env.reaction_action_spaces.values()
            for subspace in space
            if subspace.action_type.is_birxn
        }
        signatures = [(), *((site,) for site in sorted(env.synthon_types))]
        self._bi_candidates: dict[tuple[int, ...], list[tuple[str, BiReaction]]] = {
            signature: [
                (name, reaction)
                for name, reaction in self.bi_reactions.items()
                if (
                    name,
                    "-".join(map(str, (reaction.attachment_type, *signature))),
                )
                in self._bi_actions
            ]
            for signature in signatures
        }

    def run(
        self,
        smiles: str,
        *,
        max_reactions: int | None = None,
        max_synthons: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        if max_reactions is None:
            max_reactions = self.max_reactions
        if max_synthons is None:
            max_synthons = self.max_synthons
        if smiles is None:
            return []
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return []

        dfs = _RetroDFS(self, max_reactions, max_synthons, prune=self.prune)
        mol = Molecule(rdmol=rdmol)
        return dfs.run(mol, known_trajectories)

    def one_step(self, smiles: str) -> list[BackwardEdge]:
        """Return reverse steps in canonical order, using only the molecule.

        No budget checks or recursion; the empty state has no parents.
        """
        if not smiles:
            return []
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return []
        dfs = _RetroDFS(self, self.max_reactions, self.max_synthons, prune=False)
        return dfs.one_step(Molecule(rdmol=rdmol))


class _RetroDFS:
    def __init__(
        self,
        worker: Worker,
        max_reactions: int,
        max_synthons: int,
        prune: bool = True,
    ):
        self.prune: int = prune
        self.max_depth = max_reactions + 1  # Includes FirstSynthon.
        self.brick_types = worker.brick_types
        self.synthon_search = worker.synthon_search
        self.uni_candidates = worker._uni_candidates
        self.bi_candidates = worker._bi_candidates
        self.bi_actions = worker._bi_actions
        # Reverse chemistry is independent of DFS budgets and known routes.
        self.cache: dict[str, dict] = {"unirxn": {}, "birxn": {}, "traj": {}}

        self._max_synthons = min(max_synthons, self.max_depth)

    def run(
        self,
        molecule: Molecule,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        if known_trajectories is not None:
            known_trajectories = [
                route
                for route in known_trajectories
                if len(route) <= self.max_depth
                and sum(not action.action_type.is_unirxn for action, _ in route)
                <= self._max_synthons
            ]
            if self.prune and known_trajectories:
                self._max_synthons = min(
                    self._max_synthons,
                    min(
                        sum(not action.action_type.is_unirxn for action, _ in route)
                        for route in known_trajectories
                    ),
                )
        return self._dfs(molecule, 1, 1, known_trajectories)

    def _dfs(
        self,
        molecule: Molecule,
        depth: int,
        num_synthons: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        # Count removed synthons plus FirstSynthon; unimolecular steps only add depth.
        if depth > self.max_depth or num_synthons > self._max_synthons:
            return []
        # Cache only under the same remaining reaction and synthon budgets.
        canonical = molecule.smiles
        key = (canonical, depth, num_synthons, self._max_synthons)
        if known_trajectories is None and key in self.cache["traj"]:
            return self.cache["traj"][key]
        trajectories = list(known_trajectories or [])
        # Keep known suffixes and skip rediscovering their first action/parent pair.
        branch_keys = {trajectory[0] for trajectory in trajectories}

        # Restore the catalog marker to check for a direct FirstSynthon origin.
        signature = typed_dummy_isotopes(molecule.rdmol)
        if len(signature) == 1:
            library_name = str(signature[0])
            if library_name in self.brick_types:
                brick = Chem.Mol(molecule.rdmol)
                for atom in brick.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetIsotope(0)
                brick_smiles = Chem.MolToSmiles(brick)
                synthon_index = self.synthon_search[library_name].get(brick_smiles)
                if synthon_index is not None:
                    if self.prune and num_synthons < self._max_synthons:
                        self._max_synthons = num_synthons
                    action = Action(
                        ActionType.FIRST_SYNTHON,
                        library_name=library_name,
                        synthon_index=synthon_index,
                    )
                    edge = (action, "")
                    if edge not in branch_keys:
                        trajectories.append([edge])
                        branch_keys.add(edge)

        if depth < self.max_depth:
            for name, reaction, action in self.uni_candidates.get(signature, ()):
                reverse_key = (name, canonical)
                products_list = self.cache["unirxn"].get(reverse_key)
                if products_list is None:
                    products_list = reaction.run_reverse(molecule)
                    self.cache["unirxn"][reverse_key] = products_list
                for products in products_list:
                    if len(products) != 1:
                        continue
                    precursor = products[0]
                    parent_smiles = precursor.smiles
                    if typed_dummy_isotopes(precursor.rdmol) != (reaction.input_type,):
                        continue
                    edge = (action, parent_smiles)
                    if edge in branch_keys:
                        continue
                    forward_product = reaction.run_forward(precursor)
                    if forward_product is None or forward_product.smiles != canonical:
                        continue
                    suffixes = self._dfs(precursor, depth + 1, num_synthons)
                    if suffixes:
                        trajectories.extend([edge, *suffix] for suffix in suffixes)
                        branch_keys.add(edge)

            for name, reaction in self.bi_candidates.get(signature, ()):
                if num_synthons >= self._max_synthons:
                    break
                reverse_key = (name, canonical)
                products_list = self.cache["birxn"].get(reverse_key)
                if products_list is None:
                    products_list = self._reverse_birxn(reaction, molecule)
                    self.cache["birxn"][reverse_key] = products_list
                for child, synthon in products_list:
                    if num_synthons >= self._max_synthons:
                        break
                    if typed_dummy_isotopes(child.rdmol) != (reaction.state_type,):
                        continue
                    # The incoming type and remaining dummy label identify the oriented
                    # library.
                    synthon_sites = typed_dummy_isotopes(synthon.rdmol)
                    if not synthon_sites or synthon_sites[0] != 0:
                        continue
                    library_name = "-".join(
                        map(str, (reaction.attachment_type, *synthon_sites[1:]))
                    )
                    if (name, library_name) not in self.bi_actions:
                        continue
                    library = self.synthon_search.get(library_name)
                    if library is None:
                        continue
                    synthon_index = library.get(synthon.smiles)
                    if synthon_index is None:
                        continue
                    reverse_action = Action(
                        ActionType.BIRXN_BRICK
                        if len(synthon_sites) == 1
                        else ActionType.BIRXN_LINKER,
                        reaction=name,
                        library_name=library_name,
                        synthon_index=synthon_index,
                    )
                    edge = (reverse_action, child.smiles)
                    if edge in branch_keys:
                        continue
                    forward_product = reaction.run_forward(child, synthon)
                    if forward_product is None or forward_product.smiles != canonical:
                        continue
                    suffixes = self._dfs(child, depth + 1, num_synthons + 1)
                    if suffixes:
                        trajectories.extend([edge, *suffix] for suffix in suffixes)
                        branch_keys.add(edge)

        # Known routes are rollout-specific; cache only unseeded searches.
        if known_trajectories is None:
            self.cache["traj"][key] = trajectories
        return trajectories

    def _reverse_birxn(
        self, reaction: BiReaction, product: Molecule, *, complete: bool = False
    ) -> list[tuple[Molecule, Molecule]]:
        """Find catalog-supported precursor pairs in canonical DFS order.

        complete=True uses the forward template's inverse for proposals.
        """
        template = reaction.inverse_reaction if complete else reaction.reverse_reaction
        products = {}
        for raw in template.RunReactants((product.rdmol,), 0):
            if len(raw) != 2 or typed_dummy_isotopes(raw[0]) != (reaction.state_type,):
                continue
            sites = typed_dummy_isotopes(raw[1])
            if not sites or sites[0] != 0:
                continue
            library_name = "-".join(map(str, (reaction.attachment_type, *sites[1:])))
            library = self.synthon_search.get(library_name)
            if library is None:
                continue

            # Reject missing catalog synthons before sanitizing the parent.
            molecules, keys = [None, None], [None, None]
            for index in (1, 0):
                molecule = raw[index]
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(molecule)
                        if molecule.HasSubstructMatch(_HYDROGEN_ATOM):
                            molecule = Chem.RemoveHs(molecule)
                        smiles = Chem.MolToSmiles(molecule)
                except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                    break
                if index == 1 and smiles not in library:
                    break
                molecules[index], keys[index] = molecule, smiles
            if any(key is None for key in keys):
                continue
            key = tuple(keys)
            if key not in products:
                products[key] = (
                    Molecule(smiles=keys[0], rdmol=molecules[0]),
                    Molecule(smiles=keys[1], rdmol=molecules[1]),
                )
        return [products[key] for key in sorted(products)]

    def one_step(self, molecule: Molecule) -> list[BackwardEdge]:
        """Return reverse candidates in canonical order, without recursion.

        Use inverse forward templates for proposals, then require a catalog synthon
        and a forward reaction that reproduces the molecule. Ambiguous forward
        reactions are rejected. Parents may be dead ends; no route search is done.

        _dfs uses the curated reverse templates for route search.
        """
        canonical = molecule.smiles
        signature = typed_dummy_isotopes(molecule.rdmol)
        edges: set[BackwardEdge] = set()

        # Check whether the molecule is itself a catalog brick.
        if len(signature) == 1:
            library_name = str(signature[0])
            if library_name in self.brick_types:
                brick = Chem.Mol(molecule.rdmol)
                for atom in brick.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetIsotope(0)
                synthon_index = self.synthon_search[library_name].get(
                    Chem.MolToSmiles(brick)
                )
                if synthon_index is not None:
                    action = Action(
                        ActionType.FIRST_SYNTHON,
                        library_name=library_name,
                        synthon_index=synthon_index,
                    )
                    edges.add((action, ""))

        for _, reaction, action in self.uni_candidates.get(signature, ()):
            for products in reaction.run_inverse(molecule):
                if len(products) != 1:
                    continue
                precursor = products[0]
                if typed_dummy_isotopes(precursor.rdmol) != (reaction.input_type,):
                    continue
                if _forward_reproduces(reaction, canonical, precursor):
                    edges.add((action, precursor.smiles))

        for name, reaction in self.bi_candidates.get(signature, ()):
            for child, synthon in self._reverse_birxn(reaction, molecule, complete=True):
                if typed_dummy_isotopes(child.rdmol) != (reaction.state_type,):
                    continue
                synthon_sites = typed_dummy_isotopes(synthon.rdmol)
                if not synthon_sites or synthon_sites[0] != 0:
                    continue
                library_name = "-".join(
                    map(str, (reaction.attachment_type, *synthon_sites[1:]))
                )
                if (name, library_name) not in self.bi_actions:
                    continue
                library = self.synthon_search.get(library_name)
                if library is None:
                    continue
                synthon_index = library.get(synthon.smiles)
                if synthon_index is None:
                    continue
                if not _forward_reproduces(reaction, canonical, child, synthon):
                    continue
                action = Action(
                    ActionType.BIRXN_BRICK
                    if len(synthon_sites) == 1
                    else ActionType.BIRXN_LINKER,
                    reaction=name,
                    library_name=library_name,
                    synthon_index=synthon_index,
                )
                edges.add((action, child.smiles))
        return sorted(edges, key=_edge_key)


def _edge_key(edge: BackwardEdge) -> tuple:
    """Sort edges consistently before caching."""
    action, parent = edge
    return (
        int(action.action_type),
        action.reaction or "",
        action.library_name or "",
        -1 if action.synthon_index is None else action.synthon_index,
        parent,
    )


def _forward_reproduces(reaction, canonical: str, *reactants: Molecule) -> bool:
    """True if running `reaction` forward on `reactants` gives the molecule `canonical`."""
    try:
        product = reaction.run_forward(*reactants)
    except ValueError:  # ambiguous template: several distinct forward products
        return False
    return product is not None and product.smiles == canonical


_WORKER: Worker | None = None


def _init_worker(worker: Worker) -> None:
    global _WORKER
    _WORKER = worker


def _worker_run(
    smiles: str,
    max_reactions: int | None,
    known_trajectories: list[BackwardTrajectory] | None,
    max_synthons: int | None,
) -> list[BackwardTrajectory]:
    assert _WORKER is not None
    return _WORKER.run(
        smiles,
        max_reactions=max_reactions,
        known_trajectories=known_trajectories,
        max_synthons=max_synthons,
    )


def _worker_one_step(smiles: str) -> list[BackwardEdge]:
    assert _WORKER is not None
    return _WORKER.one_step(smiles)


class RetroSynthesisAnalyzer:
    """Search routes locally or in worker processes.

    Limits default to the environment's generation maxima. Results are
    reverse-ordered, ending with FirstSynthon. Search favors fewer synthons
    and may omit routes within the limits.

    one_step_many returns molecule-only candidates without a route search.
    """

    def __init__(
        self,
        env: SynthesisEnv,
        workers: int = 0,
        prune: bool = True,
        one_step_cache_size: int = 200_000,
    ):
        self.worker = Worker(env, prune=prune)
        self._workers = workers
        self._one_step_cache: OrderedDict[str, tuple[BackwardEdge, ...]] = OrderedDict()
        self._one_step_cache_size = one_step_cache_size
        self.pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(self.worker,),
            )
            if workers > 0
            else None
        )
        self.futures: list[tuple[int, Future[list[BackwardTrajectory]]]] = []
        self.results: list[tuple[int, list[BackwardTrajectory]]] = []

    def run(
        self,
        smiles: str,
        max_reactions: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
        *,
        max_synthons: int | None = None,
    ) -> list[BackwardTrajectory]:
        if self.pool is None:
            return self.worker.run(
                smiles,
                max_reactions=max_reactions,
                known_trajectories=known_trajectories,
                max_synthons=max_synthons,
            )
        return self.pool.submit(
            _worker_run, smiles, max_reactions, known_trajectories, max_synthons
        ).result()

    def submit(
        self,
        key: int,
        smiles: str,
        max_reactions: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
        *,
        max_synthons: int | None = None,
    ) -> None:
        if self.pool is None:
            self.results.append(
                (
                    key,
                    self.worker.run(
                        smiles,
                        max_reactions=max_reactions,
                        known_trajectories=known_trajectories,
                        max_synthons=max_synthons,
                    ),
                )
            )
        else:
            future = self.pool.submit(
                _worker_run, smiles, max_reactions, known_trajectories, max_synthons
            )
            self.futures.append((key, future))

    def one_step(self, smiles: str) -> tuple[BackwardEdge, ...]:
        return self.one_step_many([smiles])[0]

    def one_step_many(self, smiles_list: Sequence[str]) -> list[tuple[BackwardEdge, ...]]:
        """Return canonical reverse candidates, cached by SMILES in an LRU.

        Cache misses are batched through the worker pool when available.
        Returned tuples are shared with the cache.
        """
        cache = self._one_step_cache
        found: dict[str, tuple[BackwardEdge, ...]] = {}
        missing: list[str] = []
        for smiles in dict.fromkeys(smiles_list):
            hit = cache.get(smiles)
            if hit is None:
                missing.append(smiles)
            else:
                cache.move_to_end(smiles)
                found[smiles] = hit
        if missing:
            if self.pool is None:
                computed = [self.worker.one_step(smiles) for smiles in missing]
            else:
                chunk = max(1, len(missing) // (4 * self._workers))
                computed = list(self.pool.map(_worker_one_step, missing, chunksize=chunk))
            for smiles, edges in zip(missing, computed, strict=True):
                found[smiles] = cache[smiles] = tuple(edges)
            while len(cache) > self._one_step_cache_size:
                cache.popitem(last=False)
        return [found[smiles] for smiles in smiles_list]

    def result(self) -> list[tuple[int, list[BackwardTrajectory]]]:
        """Wait for pending searches and drain their results in submission order."""
        if self.pool is None:
            results = self.results
            self.results = []
            return results
        results = [(key, future.result()) for key, future in self.futures]
        self.futures = []
        return results

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
