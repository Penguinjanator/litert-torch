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
"""Unit tests for sampling_executor."""

from unittest import mock
from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.calib import sampling_executor
import numpy as np


def _create_mock_runner(
    input_names: list[str],
    output_dict: dict[str, np.ndarray] | None = None,
):
  """Helper to create a mock signature runner."""
  runner = mock.MagicMock()
  runner.get_input_details.return_value = {
      name: {
          'name': name,
          'index': i,
          'shape': np.array([1, 4]),
          'dtype': np.float32,
          'quantization_parameters': {'scales': []},
      }
      for i, name in enumerate(input_names)
  }
  runner.get_output_details.return_value = {}
  outputs = output_dict if output_dict is not None else {}
  runner.side_effect = lambda **kwargs: {**outputs}
  return runner


class SamplingExecutorTest(absltest.TestCase):

  def test_try_get_quantized_input_raises_on_missing_key(self):
    runner = _create_mock_runner(['k_cache_0', 'v_cache_0', 'valid_mask'])
    inputs = {
        'k_cache_0': np.zeros((1, 4, 16), dtype=np.float32),
        'v_cache_0': np.zeros((1, 4, 16), dtype=np.float32),
    }
    with self.assertRaisesRegex(
        ValueError, "Missing required signature input 'valid_mask'"
    ):
      sampling_executor.try_get_quantized_input(inputs, runner)

  def test_try_get_quantized_input_success_when_all_keys_provided(self):
    runner = _create_mock_runner(['k_cache_0', 'v_cache_0', 'valid_mask'])
    inputs = {
        'k_cache_0': np.zeros((1, 4, 16), dtype=np.float32),
        'v_cache_0': np.zeros((1, 4, 16), dtype=np.float32),
        'valid_mask': np.ones((1, 4), dtype=bool),
    }
    result = sampling_executor.try_get_quantized_input(inputs, runner)
    self.assertIn('valid_mask', result)
    self.assertIn('k_cache_0', result)

  def test_prefill_chunk_passes_valid_mask_when_declared(self):
    mock_prefill_runner = _create_mock_runner(
        ['embeddings', 'rope', 'mask', 'k_cache_0'],
        output_dict={
            'k_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
            'v_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
        },
    )
    mock_mask_runner = _create_mock_runner(
        ['time_step', 'input_tokens', 'valid_mask'],
        output_dict={'mask': np.zeros((1, 4, 4), dtype=np.float32)},
    )
    mock_rope_runner = _create_mock_runner(
        ['input_pos'],
        output_dict={'rope': np.zeros((1, 4, 16), dtype=np.float32)},
    )
    mock_cache_update_runner = _create_mock_runner(
        ['k_slice_0', 'v_slice_0', 'k_cache_0', 'input_pos', 'valid_mask'],
        output_dict={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
    )

    executor = object.__new__(sampling_executor.Executor)
    executor.stream_output = False
    executor.cache_length = 128
    executor.prefill_runners = {4: mock_prefill_runner}
    executor.prefill_mask_runners = {4: mock_mask_runner}
    executor.prefill_rope_runners = {4: mock_rope_runner}
    executor.prefill_per_layer_embedder_runners = None
    executor.prefill_cache_update_runners = {4: mock_cache_update_runner}

    decode_state = sampling_executor.DecodeState(
        kv_cache={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
        num_input_tokens=3,
        token_ids=np.array([[10, 20, 30]], dtype=np.int32),
        sampled_tokens=np.array([[]], dtype=np.int32),
        logits=None,
        time_step=0,
        generate=False,
        done=False,
        processed_embeds=np.zeros((1, 4, 16), dtype=np.float32),
    )

    updated_state = executor.prefill_chunk(decode_state)

    self.assertIsNotNone(updated_state)
    # Verify valid_mask in mask runner
    mask_called_kwargs = mock_mask_runner.call_args[1]
    self.assertIn('valid_mask', mask_called_kwargs)
    np.testing.assert_array_equal(
        mask_called_kwargs['valid_mask'],
        np.array([[True, True, True, False]]),
    )
    # Verify valid_mask in cache update runner
    called_kwargs = mock_cache_update_runner.call_args[1]
    self.assertIn('valid_mask', called_kwargs)
    np.testing.assert_array_equal(
        called_kwargs['valid_mask'],
        np.array([[True, True, True, False]]),
    )

  def test_prefill_chunk_omits_valid_mask_when_not_declared(self):
    mock_prefill_runner = _create_mock_runner(
        ['embeddings', 'rope', 'mask', 'k_cache_0'],
        output_dict={
            'k_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
            'v_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
        },
    )
    mock_mask_runner = _create_mock_runner(
        ['time_step', 'input_tokens'],
        output_dict={'mask': np.zeros((1, 4, 4), dtype=np.float32)},
    )
    mock_rope_runner = _create_mock_runner(
        ['input_pos'],
        output_dict={'rope': np.zeros((1, 4, 16), dtype=np.float32)},
    )
    mock_cache_update_runner = _create_mock_runner(
        ['k_slice_0', 'v_slice_0', 'k_cache_0', 'input_pos'],
        output_dict={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
    )

    executor = object.__new__(sampling_executor.Executor)
    executor.stream_output = False
    executor.cache_length = 128
    executor.prefill_runners = {4: mock_prefill_runner}
    executor.prefill_mask_runners = {4: mock_mask_runner}
    executor.prefill_rope_runners = {4: mock_rope_runner}
    executor.prefill_per_layer_embedder_runners = None
    executor.prefill_cache_update_runners = {4: mock_cache_update_runner}

    decode_state = sampling_executor.DecodeState(
        kv_cache={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
        num_input_tokens=3,
        token_ids=np.array([[10, 20, 30]], dtype=np.int32),
        sampled_tokens=np.array([[]], dtype=np.int32),
        logits=None,
        time_step=0,
        generate=False,
        done=False,
        processed_embeds=np.zeros((1, 4, 16), dtype=np.float32),
    )

    updated_state = executor.prefill_chunk(decode_state)

    self.assertIsNotNone(updated_state)
    mask_called_kwargs = mock_mask_runner.call_args[1]
    self.assertNotIn('valid_mask', mask_called_kwargs)
    called_kwargs = mock_cache_update_runner.call_args[1]
    self.assertNotIn('valid_mask', called_kwargs)

  def test_prefill_chunk_multi_chunk_all_true_valid_mask(self):
    mock_prefill_runner = _create_mock_runner(
        ['embeddings', 'rope', 'mask', 'k_cache_0'],
        output_dict={
            'k_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
            'v_slice_0': np.zeros((1, 4, 16), dtype=np.float32),
        },
    )
    mock_mask_runner = _create_mock_runner(
        ['time_step', 'input_tokens', 'valid_mask'],
        output_dict={'mask': np.zeros((1, 4, 4), dtype=np.float32)},
    )
    mock_rope_runner = _create_mock_runner(
        ['input_pos'],
        output_dict={'rope': np.zeros((1, 4, 16), dtype=np.float32)},
    )
    mock_cache_update_runner = _create_mock_runner(
        ['k_slice_0', 'v_slice_0', 'k_cache_0', 'input_pos', 'valid_mask'],
        output_dict={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
    )

    executor = object.__new__(sampling_executor.Executor)
    executor.stream_output = False
    executor.cache_length = 128
    executor.prefill_runners = {4: mock_prefill_runner}
    executor.prefill_mask_runners = {4: mock_mask_runner}
    executor.prefill_rope_runners = {4: mock_rope_runner}
    executor.prefill_per_layer_embedder_runners = None
    executor.prefill_cache_update_runners = {4: mock_cache_update_runner}

    # 6 total input tokens, chunk size is 4 -> remaining (6) > max runner (4)
    decode_state = sampling_executor.DecodeState(
        kv_cache={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
        num_input_tokens=6,
        token_ids=np.array([[10, 20, 30, 40, 50, 60]], dtype=np.int32),
        sampled_tokens=np.array([[]], dtype=np.int32),
        logits=None,
        time_step=0,
        generate=False,
        done=False,
        processed_embeds=np.zeros((1, 6, 16), dtype=np.float32),
    )

    updated_state = executor.prefill_chunk(decode_state)

    self.assertIsNotNone(updated_state)
    self.assertEqual(updated_state.time_step, 4)
    self.assertFalse(updated_state.generate)
    # Mask runner gets all-True mask [1, 4]
    mask_called_kwargs = mock_mask_runner.call_args[1]
    self.assertIn('valid_mask', mask_called_kwargs)
    np.testing.assert_array_equal(
        mask_called_kwargs['valid_mask'],
        np.array([[True, True, True, True]]),
    )
    # Cache update runner gets all-True mask [1, 4]
    called_kwargs = mock_cache_update_runner.call_args[1]
    self.assertIn('valid_mask', called_kwargs)
    np.testing.assert_array_equal(
        called_kwargs['valid_mask'],
        np.array([[True, True, True, True]]),
    )

  def test_decode_step_passes_valid_mask_when_declared(self):
    mock_mask_runner = _create_mock_runner(
        ['time_step', 'input_tokens', 'valid_mask'],
        output_dict={'mask': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_embedder_runner = _create_mock_runner(
        ['token_ids'],
        output_dict={'embeddings': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_rope_runner = _create_mock_runner(
        ['input_pos'],
        output_dict={'rope': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_decode_runner = _create_mock_runner(
        ['embeddings', 'rope', 'mask', 'k_cache_0'],
        output_dict={
            'logits': np.zeros((1, 1, 32000), dtype=np.float32),
            'k_slice_0': np.zeros((1, 1, 16), dtype=np.float32),
            'v_slice_0': np.zeros((1, 1, 16), dtype=np.float32),
        },
    )
    mock_cache_update_runner = _create_mock_runner(
        ['k_slice_0', 'v_slice_0', 'k_cache_0', 'input_pos', 'valid_mask'],
        output_dict={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
    )

    executor = object.__new__(sampling_executor.Executor)
    executor.stream_output = False
    executor.cache_length = 128
    executor.decode_mask_runner = mock_mask_runner
    executor.decode_embedder_runner = mock_embedder_runner
    executor.decode_per_layer_embedder_runner = None
    executor.decode_rope_runner = mock_rope_runner
    executor.decode_runner = mock_decode_runner
    executor.decode_cache_update_runner = mock_cache_update_runner
    executor.sample_logits = lambda logits: np.array([[42]], dtype=np.int32)
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.stop_token_ids = ()
    mock_tokenizer.eos_id = 1
    mock_tokenizer.detokenize_internal.return_value = 'test_output'
    executor.tokenizer = mock_tokenizer
    mock_config = mock.MagicMock()
    mock_config.stop_tokens = None
    mock_config.stop_token = None
    mock_config.early_terminate_suffix = None
    executor.config = mock_config

    decode_state = sampling_executor.DecodeState(
        kv_cache={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
        num_input_tokens=3,
        token_ids=np.array([[10, 20, 30]], dtype=np.int32),
        sampled_tokens=np.array([[10, 20]], dtype=np.int32),
        logits=None,
        time_step=2,
        generate=True,
        done=False,
        next_decode_token=np.array([[30]], dtype=np.int32),
        processed_embeds=np.zeros((1, 3, 16), dtype=np.float32),
    )

    updated_state = executor.decode_step(decode_state)

    self.assertIsNotNone(updated_state)
    mask_called_kwargs = mock_mask_runner.call_args[1]
    self.assertIn('valid_mask', mask_called_kwargs)
    np.testing.assert_array_equal(
        mask_called_kwargs['valid_mask'],
        np.array([[True]]),
    )
    called_kwargs = mock_cache_update_runner.call_args[1]
    self.assertIn('valid_mask', called_kwargs)
    np.testing.assert_array_equal(
        called_kwargs['valid_mask'],
        np.array([[True]]),
    )

  def test_decode_step_omits_valid_mask_when_not_declared(self):
    mock_mask_runner = _create_mock_runner(
        ['time_step', 'input_tokens'],
        output_dict={'mask': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_embedder_runner = _create_mock_runner(
        ['token_ids'],
        output_dict={'embeddings': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_rope_runner = _create_mock_runner(
        ['input_pos'],
        output_dict={'rope': np.zeros((1, 1, 16), dtype=np.float32)},
    )
    mock_decode_runner = _create_mock_runner(
        ['embeddings', 'rope', 'mask', 'k_cache_0'],
        output_dict={
            'logits': np.zeros((1, 1, 32000), dtype=np.float32),
            'k_slice_0': np.zeros((1, 1, 16), dtype=np.float32),
            'v_slice_0': np.zeros((1, 1, 16), dtype=np.float32),
        },
    )
    mock_cache_update_runner = _create_mock_runner(
        ['k_slice_0', 'v_slice_0', 'k_cache_0', 'input_pos'],
        output_dict={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
    )

    executor = object.__new__(sampling_executor.Executor)
    executor.stream_output = False
    executor.cache_length = 128
    executor.decode_mask_runner = mock_mask_runner
    executor.decode_embedder_runner = mock_embedder_runner
    executor.decode_per_layer_embedder_runner = None
    executor.decode_rope_runner = mock_rope_runner
    executor.decode_runner = mock_decode_runner
    executor.decode_cache_update_runner = mock_cache_update_runner
    executor.sample_logits = lambda logits: np.array([[42]], dtype=np.int32)
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.stop_token_ids = ()
    mock_tokenizer.eos_id = 1
    mock_tokenizer.detokenize_internal.return_value = 'test_output'
    executor.tokenizer = mock_tokenizer
    mock_config = mock.MagicMock()
    mock_config.stop_tokens = None
    mock_config.stop_token = None
    mock_config.early_terminate_suffix = None
    executor.config = mock_config

    decode_state = sampling_executor.DecodeState(
        kv_cache={
            'k_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
            'v_cache_0': np.zeros((1, 16, 16), dtype=np.float32),
        },
        num_input_tokens=3,
        token_ids=np.array([[10, 20, 30]], dtype=np.int32),
        sampled_tokens=np.array([[10, 20]], dtype=np.int32),
        logits=None,
        time_step=2,
        generate=True,
        done=False,
        next_decode_token=np.array([[30]], dtype=np.int32),
        processed_embeds=np.zeros((1, 3, 16), dtype=np.float32),
    )

    updated_state = executor.decode_step(decode_state)

    self.assertIsNotNone(updated_state)
    mask_called_kwargs = mock_mask_runner.call_args[1]
    self.assertNotIn('valid_mask', mask_called_kwargs)
    called_kwargs = mock_cache_update_runner.call_args[1]
    self.assertNotIn('valid_mask', called_kwargs)


if __name__ == '__main__':
  absltest.main()
