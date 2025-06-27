from __future__ import annotations

from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

if TYPE_CHECKING:
    from typing import Any

    from torch.utils.hooks import RemovableHandle

    from stable_baselines3.common.type_aliases import TensorDict


def hierarchical_summary(model: nn.Module, print_summary: bool = False) -> tuple[str, int]:
    """
    Source: https://github.com/amarczew/pytorch_model_summary
    """

    def repr(model: nn.Module) -> tuple[str, int]:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            if module is None:
                continue
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)  # type: ignore[no-untyped-call]
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("  # type: ignore[no-untyped-call]
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        main_str += f", {total_params:,} params"
        return main_str, total_params

    string, count = repr(model)

    # Building hierarchical output
    _pad = int(max(max(len(_) for _ in string.split("\n")) - 20, 0) / 2)
    lines = list()
    lines.append("=" * _pad + " Hierarchical Summary " + "=" * _pad + "\n")
    lines.append(string)
    lines.append("\n\n" + "=" * (_pad * 2 + 22) + "\n")

    str_summary = "\n".join(lines)
    if print_summary:
        print(str_summary)

    return str_summary, count


def summary(
    model: nn.Module,
    *inputs: torch.Tensor | TensorDict,
    batch_size: int = -1,
    show_input: bool = False,
    show_hierarchical: bool = False,
    print_summary: bool = False,
    max_depth: int | None = 1,
    show_parent_layers: bool = False,
) -> str:
    """
    Adapted from https://github.com/amarczew/pytorch_model_summary
    """

    max_depth = max_depth if max_depth is not None else 9999

    # create properties
    summary: dict[str, Any] = OrderedDict()
    module_summary: dict[int, Any] = dict()
    module_mapped = set()
    hooks: list[RemovableHandle] = []

    def build_module_tree(module: nn.Module) -> None:
        def _in(module: nn.Module, id_parent: int, depth: int) -> None:
            for c in module.children():
                # ModuleList and Sequential do not count as valid layers
                if isinstance(c, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                    _in(c, id_parent, depth)
                else:
                    _module_name = str(c.__class__).split(".")[-1].split("'")[0]
                    _parent_layers = (
                        f"{module_summary[id_parent].get('parent_layers')}"
                        f"{'/' if module_summary[id_parent].get('parent_layers') != '' else ''}"
                        f"{module_summary[id_parent]['module_name']}"
                    )

                    module_summary[id(c)] = {
                        "module_name": _module_name,
                        "parent_layers": _parent_layers,
                        "id_parent": id_parent,
                        "depth": depth,
                        "n_children": 0,
                    }

                    module_summary[id_parent]["n_children"] += 1

                    _in(c, id(c), depth + 1)

        # Defining summary for the main module
        module_summary[id(module)] = {
            "module_name": str(module.__class__).split(".")[-1].split("'")[0],
            "parent_layers": "",
            "id_parent": None,
            "depth": 0,
            "n_children": 0,
        }

        _in(module, id_parent=id(module), depth=1)

        # Defining layers that will be printed
        for k, v in module_summary.items():
            module_summary[k]["show"] = v["depth"] == max_depth or (v["depth"] < max_depth and v["n_children"] == 0)

    def register_hook(module: nn.Module) -> None:
        def shapes(x: Any) -> list[list[int]]:
            _lst = list()

            def _shapes(_: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor] | dict[Any, torch.Tensor]) -> None:
                if isinstance(_, torch.Tensor):
                    _lst.append(list(_.size()))
                elif isinstance(_, (tuple, list)):
                    for _x in _:
                        _shapes(_x)
                elif isinstance(_, dict):
                    for _v in _.values():
                        _shapes(_v)
                else:
                    # TODO: decide what to do when there is an input which is not a tensor
                    raise Exception("Object not supported")

            _shapes(x)

            return _lst

        def hook(module: nn.Module, input: torch.Tensor | TensorDict, output: torch.Tensor | TensorDict | None = None) -> None:
            if id(module) in module_mapped:
                return

            module_mapped.add(id(module))
            module_name = module_summary.get(id(module)).get("module_name")  # type: ignore[union-attr]
            module_idx = len(summary)

            m_key = "%s-%i" % (module_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["parent_layers"] = module_summary.get(id(module)).get("parent_layers")  # type: ignore[union-attr]

            summary[m_key]["input_shape"] = shapes(input) if len(input) != 0 else input

            if show_input is False and output is not None:
                summary[m_key]["output_shape"] = shapes(output)

            params = torch.tensor(0)
            params_trainable = torch.tensor(0)
            trainable = False
            for m in module.parameters():
                _params = torch.prod(torch.LongTensor(list(m.size())))
                params += _params
                params_trainable += _params if m.requires_grad else 0
                # if any parameter is trainable, then module is trainable
                trainable = trainable or m.requires_grad

            summary[m_key]["nb_params"] = params
            summary[m_key]["nb_params_trainable"] = params_trainable
            summary[m_key]["trainable"] = trainable

        _map_module = module_summary.get(id(module), None)
        if _map_module is not None and _map_module.get("show"):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))  # type: ignore[arg-type]
            else:
                hooks.append(module.register_forward_hook(hook))  # type: ignore[arg-type]

    # register id of parent modules
    build_module_tree(model)

    # register hook
    model.apply(register_hook)

    model_training = model.training

    model.eval()
    model(*inputs)

    if model_training:
        model.train()

    # remove these hooks
    for h in hooks:
        h.remove()

    # params to format output - dynamic width
    _key_shape = "input_shape" if show_input else "output_shape"
    _len_str_parent = max([len(v["parent_layers"]) for v in summary.values()] + [13]) + 3
    _len_str_layer = max([len(layer) for layer in summary.keys()] + [15]) + 3
    _len_str_shapes = max([len(", ".join([str(_) for _ in summary[layer][_key_shape]])) for layer in summary] + [15]) + 3
    _len_line = 35 + _len_str_parent * int(show_parent_layers) + _len_str_layer + _len_str_shapes
    fmt = ("{:>%d} " % _len_str_parent if show_parent_layers else "") + "{:>%d}  {:>%d} {:>15} {:>15}" % (_len_str_layer, _len_str_shapes)

    """ starting to build output text """

    # Table header
    lines = list()
    lines.append("-" * _len_line)
    _fmt_args = ("Parent Layers",) if show_parent_layers else ()
    _fmt_args += ("Layer (type)", f"{'Input' if show_input else 'Output'} Shape", "Param #", "Tr. Param #")  # type: ignore[assignment]
    lines.append(fmt.format(*_fmt_args))
    lines.append("=" * _len_line)

    total_params = 0
    trainable_params = 0
    for layer in summary:
        # Table content (for each layer)
        _fmt_args = (summary[layer]["parent_layers"],) if show_parent_layers else ()
        _fmt_args += (  # type: ignore[assignment]
            layer,
            ", ".join([str(_) for _ in summary[layer][_key_shape]]),
            "{:,}".format(summary[layer]["nb_params"]),
            "{:,}".format(summary[layer]["nb_params_trainable"]),
        )
        line_new = fmt.format(*_fmt_args)
        lines.append(line_new)

        total_params += summary[layer]["nb_params"]
        trainable_params += summary[layer]["nb_params_trainable"]

    # Table footer
    lines.append("=" * _len_line)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append(f"Non-trainable params: {total_params - trainable_params:,}")
    if batch_size != -1:
        lines.append(f"Batch size: {batch_size:,}")
    lines.append("-" * _len_line)

    if show_hierarchical:
        h_summary, _ = hierarchical_summary(model, print_summary=False)
        lines.append("\n")
        lines.append(h_summary)

    str_summary = "\n".join(lines)
    if print_summary:
        print(str_summary)

    return str_summary


class ModelSummaryCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _init_callback(self) -> None:
        obs = self.model.observation_space.sample()
        if isinstance(obs, dict):
            obs_unsqueezed = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        else:
            obs_unsqueezed = np.expand_dims(obs, axis=0)  # type: ignore[assignment]

        print(summary(self.model.policy, obs_as_tensor(obs_unsqueezed, self.model.device), max_depth=None))

    def _on_step(self) -> bool:
        return True
