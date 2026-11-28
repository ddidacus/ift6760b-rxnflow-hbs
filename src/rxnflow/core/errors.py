"""Exceptions shared by the synthesis environment and GFlowNet policy."""


class InvalidTransition(ValueError):
    """A selected action is unavailable or has no valid product."""


class MissingReverseEdge(ValueError):
    """`one_step` did not return the edge that produced a state.

    P_B is a distribution over the one-step reverse candidates of a state, so the edge the
    forward policy took must be one of them. When it is not, the step cannot be scored and
    a backward walk could never reproduce it: the candidate generator is at fault.
    """

    def __init__(self, product_smiles: str, action: object, parent: str):
        super().__init__(
            f"one_step({product_smiles!r}) did not return the generating edge "
            f"{action} from parent {parent!r}"
        )
        self.product_smiles = product_smiles
        self.action = action
        self.parent = parent
