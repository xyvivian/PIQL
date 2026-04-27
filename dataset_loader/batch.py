from abc import ABCMeta, abstractmethod
from typing import Set, Optional
from dataclasses import dataclass, fields
import torch
from torch.utils.data import DataLoader
from functools import reduce
from typing import List
from dataclasses import dataclass, field

@dataclass
class Batch:
    """
    A batch of data, with non-optional x, y, and target_y attributes.
    All other attributes are optional.

    If you want to add an attribute for testing only, you can just assign it after creation like:
    ```
        batch = Batch(x=x, y=y, target_y=target_y)
        batch.test_attribute = test_attribute
    ```
    """
    # Required entries
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor
    model_names: List[str] = field(default_factory=list)

    # Optional Batch Entries
    style: Optional[torch.Tensor] = None
    internal_xs: Optional[torch.Tensor] = None
    style_hyperparameter_values: Optional[torch.Tensor] = None
    single_eval_pos: Optional[torch.Tensor] = None
    causal_model_dag: Optional[object] = None
    mean_prediction: Optional[bool] = None  # this controls whether to do mean prediction in bar_distribution for nonmyopic BO

    def other_filled_attributes(self, set_of_attributes: Set[str] = frozenset(('x', 'y', 'target_y'))):
        return [f.name for f in fields(self)
                if f.name not in set_of_attributes and
                getattr(self, f.name) is not None]


def safe_merge_batches_in_batch_dim(*batches, ignore_attributes=[]):
    """
    Merge all supported non-None fields in a pre-specified (general) way,
    e.g. mutliple batch.x are concatenated in the batch dimension.
    :param ignore_attributes: attributes to remove from the merged batch, treated as if they were None.
    :return:
    """
    not_none_fields = [f.name for f in fields(batches[0]) if f.name not in ignore_attributes and getattr(batches[0], f.name) is not None]
    assert all([set(not_none_fields) == set([f.name for f in fields(b) if f.name not in ignore_attributes and getattr(b, f.name) is not None]) for b in batches]), 'All batches must have the same fields!'
    merge_funcs = {
        'x': lambda xs: torch.cat(xs, 1),
        'y': lambda ys: torch.cat(ys, 1),
        'target_y': lambda target_ys: torch.cat(target_ys, 1),
        'model_names': lambda model_names: reduce(lambda a, b: a + b, model_names),
        'style': lambda styles: torch.cat(styles, 0),
    }
    assert all(f in merge_funcs for f in not_none_fields), f'Unknown fields encountered in `safe_merge_batches_in_batch_dim`.'
    return Batch(**{f: merge_funcs[f]([getattr(batch, f) for batch in batches]) for f in not_none_fields})



class PriorDataLoader(DataLoader, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, num_steps, batch_size, eval_pos_seq_len_sampler, seq_len_maximum, device, **kwargs):
        """

        :param num_steps: int, first argument, the number of steps to take per epoch, i.e. iteration of the DataLoader
        :param batch_size: int, number of datasets per batch
        :param eval_pos_seq_len_sampler: callable, it takes no arguments and returns a tuple (single eval pos, bptt)
        :param kwargs: for future compatibility it is good to have a final all catch, as new kwargs might be introduced
        """
        pass