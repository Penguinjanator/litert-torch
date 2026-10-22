# Copyright 2024 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FX graph utilities."""

from packaging import version
import torch
from torch.fx import traceback


def remove_dangling_args(graph_module: torch.fx.GraphModule):
  """Removes dangling args from the graph."""
  for node in graph_module.graph.nodes:
    if node.op == "placeholder" and not node.users:
      graph_module.graph.erase_node(node)

  graph_module.graph.lint()
  graph_module.recompile()
  return graph_module


def remove_assert_tensor_metadata_nodes(graph_module: torch.fx.GraphModule):
  """Removes aten._assert_tensor_metadata nodes from the graph.

  This op is inserted by torch.export to check tensor metadata on custom ops. It
  can break patten matching and lowering.
  """
  for node in graph_module.graph.nodes:
    if node.target == torch.ops.aten._assert_tensor_metadata.default:
      graph_module.graph.erase_node(node)

  graph_module.graph.lint()
  graph_module.recompile()
  return graph_module


def is_torch_version_under(torch_version: str) -> bool:
  """Checks if the current torch version is under the given version."""
  if not torch_version:
    raise ValueError("torch_version cannot be empty.")
  current_version = version.parse(torch.__version__)
  compared_version = version.parse(torch_version)
  return current_version < compared_version


def remove_unused_non_user_inputs(
    ep: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
  """Removes unused parameter, buffer, and constant placeholders."""
  graph_signature = ep.graph_signature
  if getattr(graph_signature, "backward_signature", None) is not None:
    return ep
  prunable_kinds = (
      torch.export.graph_signature.InputKind.PARAMETER,
      torch.export.graph_signature.InputKind.BUFFER,
      torch.export.graph_signature.InputKind.CONSTANT_TENSOR,
  )
  mutated_targets = set(graph_signature.buffers_to_mutate.values()) | set(
      graph_signature.parameters_to_mutate.values()
  )
  placeholder_nodes = [n for n in ep.graph.nodes if n.op == "placeholder"]
  if len(placeholder_nodes) != len(graph_signature.input_specs):
    return ep

  kept_specs = []
  removed_targets = set()
  removed = False
  for node, spec in zip(placeholder_nodes, graph_signature.input_specs):
    if (
        spec.kind in prunable_kinds
        and not node.users
        and spec.target not in mutated_targets
    ):
      ep.graph.erase_node(node)
      if isinstance(spec.target, str):
        removed_targets.add(spec.target)
      removed = True
    else:
      kept_specs.append(spec)

  if removed:
    graph_signature.input_specs = kept_specs
    kept_targets = {
        s.target
        for s in kept_specs
        if s.kind in prunable_kinds and isinstance(s.target, str)
    }
    for submod in ep.graph_module.modules():
      if isinstance(submod, torch.fx.GraphModule):
        kept_targets.update(
            n.target for n in submod.graph.nodes if n.op == "get_attr"
        )
    for target in removed_targets - kept_targets:
      ep.state_dict.pop(target, None)
      ep.constants.pop(target, None)
    ep.graph.lint()
    ep.graph_module.recompile()
  return ep


def reset_from_node_meta(ep: torch.export.ExportedProgram):
  """Resets the "from_node" meta field to fx node name only for the exported program."""

  for node in ep.graph.nodes:
    if not hasattr(node, "meta") or "from_node" not in node.meta:
      continue
    if is_torch_version_under("2.6.0.dev0"):
      # For torch version under 2.6.0, the history stack is a list of tuple. We
      # will only keep the current node's name in the history stack.
      history = [(node.name,)]
    else:
      # Clean up the history stack by keeping only the current node info (fx
      # node name and graph id) in a list of size 1. Clear the "from_node" field
      # to prevent redundant additions to the history stack.
      history = [traceback.NodeSource(node)]
      history[0].from_node = []
    node.meta["from_node"] = history
  return ep
