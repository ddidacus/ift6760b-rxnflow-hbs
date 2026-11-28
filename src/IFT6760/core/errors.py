# Source: src/rxnflow/core/errors.py
"""Exceptions shared by the synthesis environment and GFlowNet policy."""


class InvalidTransition(ValueError):
    """A selected action is unavailable or has no valid product."""


class MissingReverseEdge(ValueError):
    """The generated forward edge is missing from the reverse candidates.

    This prevents P_B from scoring or sampling that step and indicates a
    candidate-generation error.
    """

    def __init__(self, product_smiles: str, action: object, parent: str):
        super().__init__(
            f"one_step({product_smiles!r}) did not return the generating edge "
            f"{action} from parent {parent!r}"
        )
        self.product_smiles = product_smiles
        self.action = action
        self.parent = parent
