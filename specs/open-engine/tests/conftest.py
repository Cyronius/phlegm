"""Prerequisites for upstream runtime and native probe tests.

Keep pure Python kernel/reference checks enabled in the Phlegm monorepo.
Only the named checks below require OFLM-only sources or a GNU-style compiler.
"""
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT
OPEN_KERNELS = REPO / 'open_kernels'
for path in (OPEN_KERNELS, OPEN_KERNELS / 'model', Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

RUNTIME_CHECKS = {
    ('test_attention_pair_integration.py', 'test_hybrid_cache_preserves_low_planes_across_blocks'),
    ('test_prefill_boundaries.py', 'test_native_boundary_and_continuation_against_independent_reference'),
    ('test_prefill_ffn_replay.py', 'test_bridge_uses_current_host_arithmetic_and_refuses_source_drift'),
    ('test_prefill_ffn_replay.py', 'test_native_bridge_norm_and_activation'),
    ('test_prefill_ffn_replay.py', 'test_bridge_rejects_invalid_norm'),
    ('test_prefill_paired_input.py', 'test_private_patch_preserves_single_dispatch_body_and_rejects_drift'),
    ('test_prefill_paired_input.py', 'test_pair_reconstructs_input_and_preserves_reused_device_output'),
    ('test_prefill_prefix_sublayers.py', 'test_patch_keeps_weighted_arithmetic_and_rejects_repeat'),
    ('test_prefill_residual_trace.py', 'test_patch_rejects_source_drift_and_preserves_paired_body'),
    ('test_prefill_sublayer_trace.py', 'test_patch_is_private_and_refuses_drift'),
    ('test_prefill_weighted_cli.py', 'test_patch_preserves_single_dispatch_and_scopes_weight_correction'),
    ('test_prefill_weighted_sublayers.py', 'test_patch_preserves_weighted_dispatch_and_refuses_reapplication'),
}

NATIVE_CHECKS = {
    ('test_prefill_paired_input.py', 'test_pair_reconstructs_input_and_preserves_reused_device_output'),
    ('test_prefill_prefix_sublayers.py', 'test_capture_real_prefix_and_selected_rows'),
    ('test_prefill_residual_trace.py', 'test_writer_causal_chunks_and_padding'),
    ('test_prefill_residual_trace.py', 'test_writer_rejects_invalid_capture'),
    ('test_prefill_residual_trace.py', 'test_writer_matches_all_three_schedules_at_block_boundary'),
    ('test_prefill_sublayer_trace.py', 'test_native_capture_selects_absolute_real_position_and_refuses_overwrite'),
    ('test_prefill_weighted_cli.py', 'test_pair_weight_sum_order_and_device_ownership'),
    ('test_prefill_weighted_sublayers.py', 'test_capture_targets_real_rows_and_rejects_duplicates'),
}


def pytest_collection_modifyitems(items):
    for item in items:
        key = (item.path.name, item.originalname or item.name)
        if key in RUNTIME_CHECKS and not (ROOT / 'src/open_qwen36/core.cpp').is_file():
            item.add_marker(pytest.mark.skip(reason='requires the OFLM C++ runtime source tree'))
        if key in NATIVE_CHECKS and not shutil.which('c++'):
            item.add_marker(pytest.mark.skip(reason='requires a GNU-style c++ compiler'))
        if key == ('test_attention_pair_integration.py', 'test_hybrid_cache_preserves_low_planes_across_blocks') and not shutil.which('g++'):
            item.add_marker(pytest.mark.skip(reason='requires g++ with OpenMP and AVX2'))
        if key == ('test_gguf_twin.py', 'test_no_hand_written_family_list_remains') and not (ROOT / 'utilities/oflm-add/oflm_add/__init__.py').is_file():
            item.add_marker(pytest.mark.skip(reason='requires the OFLM model installer source'))
