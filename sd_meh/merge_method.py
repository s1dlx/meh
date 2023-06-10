import dataclasses
from enum import Enum
from typing import Set, Callable, Dict, List

from torch import Tensor


class MergeSpace(Enum):
    FULL = 'full'
    DIFFERENCE = 'difference'
    DISTRIBUTION = 'distribution'


def merge_method(spaces: MergeSpace | Set[MergeSpace]):
    if not isinstance(spaces, Set):
        spaces = {spaces}

    def decorator(callback: Callable):
        return MergeMethod(callback, spaces)

    return decorator


@dataclasses.dataclass
class MergeMethod:
    callback: Callable
    spaces: Set[MergeSpace]

    def __call__(self, hyperparams: Dict, thetas: Dict, theta_key: str):
        callback_kwargs = {[k[k.find('_')+1:]]: v[theta_key] for k, v in thetas}
        callback_kwargs.update(hyperparams)
        return self.callback(**callback_kwargs)


def schedule_space_conversions(
    inputs: List[MergeSpace],
    intermediate_options: Set[MergeSpace],
    target: MergeSpace,
):
    if target in intermediate_options:
        input_conversions = [target] * len(inputs)
    else:
        input_conversions = []
