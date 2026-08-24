# Copyright 2026 The LiteRT Torch Authors.
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
"""Pass to keep view-chain intermediates within rank 4 for the GPU delegate.

`torch.nn.functional._in_projection_packed` (used by `nn.MultiheadAttention` /
`F.scaled_dot_product_attention`) unpacks the packed QKV projection with a
`unsqueeze -> permute -> squeeze` view chain that transiently materializes
rank-5 tensors, even though the inserted dimension is size 1 and is removed
again right away. The ML Drift GPU delegate caps tensors at rank 4, so these
transient rank-5 RESHAPE/TRANSPOSE intermediates push the whole attention block
off the GPU.

These three ops are pure metadata views, so the chain
`squeeze(permute(unsqueeze(x, d), perm), sq_dims)` is exactly equal to a single
`permute` of `x` (optionally followed by a `squeeze` of any other unit dims),
which never exceeds the rank of `x`. This pass performs that algebraic folding.
It is a numerical no-op and only fires when the `unsqueeze` actually inflates a
tensor past rank 4, so graphs that are already GPU-clean are left untouched.
"""

from litert_torch import fx_infra
import torch

aten = torch.ops.aten

# Debug-info meta fields that map a node back to its source model code.
_PROVENANCE_META_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "from_node",
)


def _normalize_dim(dim: int, rank: int) -> int:
  return dim % rank


def _copy_provenance_meta(src: torch.fx.Node, dst: torch.fx.Node):
  for key in _PROVENANCE_META_KEYS:
    if key in src.meta:
      dst.meta[key] = src.meta[key]


def _squeeze_dims(squeeze_node: torch.fx.Node, rank: int) -> list[int]:
  """Returns the normalized dims removed by a squeeze node."""
  if squeeze_node.target == aten.squeeze.default:
    # squeeze() with no dim removes every size-1 dim.
    shape = squeeze_node.args[0].meta["val"].shape
    return [i for i, s in enumerate(shape) if s == 1]
  if squeeze_node.target == aten.squeeze.dim:
    return [_normalize_dim(squeeze_node.args[1], rank)]
  # aten.squeeze.dims
  return [_normalize_dim(d, rank) for d in squeeze_node.args[1]]


def _fold_unsqueeze_permute_squeeze(
    graph_module: torch.fx.GraphModule, unsqueeze_node: torch.fx.Node
) -> bool:
  """Folds `squeeze(permute(unsqueeze(x, d), perm), sq)` into a plain permute.

  Returns True if the graph was rewritten.
  """
  u_val = unsqueeze_node.meta.get("val")
  if u_val is None or u_val.dim() <= 4:
    # Only intervene when the unsqueeze inflates a tensor past the rank-4 cap.
    return False
  if len(unsqueeze_node.users) != 1:
    return False

  permute_node = next(iter(unsqueeze_node.users))
  if (
      permute_node.target != aten.permute.default
      or len(permute_node.users) != 1
  ):
    return False

  squeeze_node = next(iter(permute_node.users))
  if squeeze_node.target not in (
      aten.squeeze.dims,
      aten.squeeze.dim,
      aten.squeeze.default,
  ):
    return False

  x = unsqueeze_node.args[0]
  rank_p = u_val.dim()  # rank of the unsqueeze / permute output = rank(x) + 1
  dim_u = _normalize_dim(unsqueeze_node.args[1], rank_p)
  perm = list(permute_node.args[1])

  # Position of the inserted size-1 dim after the permute.
  inserted_pos = perm.index(dim_u)

  sq_dims = _squeeze_dims(squeeze_node, rank_p)
  if inserted_pos not in sq_dims:
    # The inserted unit dim survives the squeeze, so we cannot cancel it.
    return False

  # Build the permutation of x's original dims: walk the permute output
  # positions, skip the inserted unit slot, and map each post-unsqueeze input
  # dim back to its position in x (every input dim except dim_u shifts down by
  # one once the unsqueeze is removed).
  new_perm = []
  for out_pos in range(rank_p):
    if out_pos == inserted_pos:
      continue
    in_dim = perm[out_pos]
    new_perm.append(in_dim if in_dim < dim_u else in_dim - 1)

  # Any other dims the squeeze removed, remapped onto the new permute output
  # (which no longer contains the inserted unit slot).
  remaining_squeeze = [
      d if d < inserted_pos else d - 1 for d in sq_dims if d != inserted_pos
  ]

  graph = graph_module.graph
  with graph.inserting_before(squeeze_node):
    new_permute = graph.call_function(aten.permute.default, (x, new_perm))
    _copy_provenance_meta(squeeze_node, new_permute)
    if remaining_squeeze:
      x_val = x.meta["val"]
      with x_val.fake_mode:
        new_permute.meta["val"] = aten.permute.default(x_val, new_perm)
      result = graph.call_function(
          aten.squeeze.dims, (new_permute, remaining_squeeze)
      )
      _copy_provenance_meta(squeeze_node, result)
    else:
      result = new_permute

  # The folded result is element-for-element identical to the squeeze output,
  # so reuse its metadata.
  result.meta["val"] = squeeze_node.meta["val"]
  if "tensor_meta" in squeeze_node.meta:
    result.meta["tensor_meta"] = squeeze_node.meta["tensor_meta"]

  squeeze_node.replace_all_uses_with(result)
  return True


class ReduceViewRankPass(fx_infra.PassBase):
  """Folds unit-dim unsqueeze/permute/squeeze view chains to stay within rank 4."""

  def call(self, graph_module: torch.fx.GraphModule):
    modified = False
    # Snapshot the nodes since the graph is mutated during iteration.
    for node in list(graph_module.graph.nodes):
      if node.op != "call_function" or node.target != aten.unsqueeze.default:
        continue
      if _fold_unsqueeze_permute_squeeze(graph_module, node):
        modified = True

    if modified:
      graph_module.graph.eliminate_dead_code()
      graph_module.graph.lint()
      graph_module.recompile()

    return fx_infra.PassResult(graph_module, modified)
