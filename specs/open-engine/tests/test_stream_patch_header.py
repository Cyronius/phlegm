"""The host patch header must compile without XRT and retain KV-slice offsets."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_row_patches_compile_and_preserve_slice_geometry(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('requires a GNU-style C++ compiler')
    source = tmp_path / 'rows.cpp'
    source.write_text(r'''
#include "stream_patch.hpp"
#include <cassert>
int main() {
    using namespace stream_patch;
    AttnGeometry g;
    g.kv_row = 256;
    g.ptab_row = 64;
    std::vector<AttnRowPatch> table = {
        {0, 0, 0, 2, 0, 16},
        {1, 1, 0x80000000u, 0},
        {2, 2, 0, 1},
        {3, 3, 0, 2, 128},
    };
    uint32_t words[4] = {};
    attn_rows_apply(words, table, 7, g);
    assert(words[0] == 9 * 64);
    assert(words[1] == (0x80000000u | 7 * 256));
    assert(words[2] == 8 * 64);
    assert(words[3] == 128);
    g.kvslice = 4;
    attn_rows_apply(words, table, 7, g);
    assert(words[0] == 12 * 16);
    assert(words[1] == (0x80000000u | 7 * 256));
    assert(words[2] == 8 * 64);
    assert(words[3] == 128);
}
''')
    harness = Path(__file__).resolve().parents[3] / 'open_kernels/harness'
    binary = tmp_path / 'rows'
    subprocess.run([compiler, '-std=c++17', '-I', str(harness), str(source),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
