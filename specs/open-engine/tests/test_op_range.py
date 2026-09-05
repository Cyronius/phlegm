# Traces: OPEN-OP-RANGE (canonical spec: specs/open-engine/spec.md)
"""A recipe asking a template for a point outside its validated set fails at
generation time with the template and the parameter named."""
from __future__ import annotations

import dataclasses

import pytest

from recipes import qwen36moe as Q
from recipes.catalogue import OpRangeError, check_buffer_args, require
from recipes.load import default_spec


def test_the_27b_is_inside_every_validated_set():
    Q.recipe(default_spec())


def test_head_dim_64_is_refused_by_name():
    spec = dataclasses.replace(default_spec(), head_dim=64, rotary_dim=64)
    with pytest.raises(OpRangeError, match=r"attn: head_dim=64 is outside the validated set \{128, 256\}"):
        Q.recipe(spec)


def test_hidden_3072_is_refused_by_the_first_template_that_cannot_take_it():
    spec = dataclasses.replace(default_spec(), hidden=3072)
    with pytest.raises(OpRangeError, match=r"ln: width=3072 is outside the validated set \{2048, 2560\}"):
        Q.recipe(spec)


def test_an_unvalidated_gemv_k_is_refused():
    with pytest.raises(OpRangeError, match=r"gemv_q4: K=3072 is outside the validated set \{2048, 2560, 4096, 9728\}"):
        require("gemv_q4", K=3072)


def test_unknown_template_and_parameter():
    with pytest.raises(OpRangeError, match="no template 'conv2d'"):
        require("conv2d", K=1)
    with pytest.raises(OpRangeError, match="no parameter 'colour'"):
        require("ln", colour=1)


def test_more_than_eight_buffer_arguments_is_refused():
    check_buffer_args("ok", ["a"] * 8)
    with pytest.raises(OpRangeError, match="9 buffer arguments"):
        check_buffer_args("too_many", ["a"] * 9)


def test_a_quant_the_gemv_cannot_read_is_refused():
    with pytest.raises(OpRangeError, match="quant='q4_k'"):
        Q.recipe(dataclasses.replace(default_spec(), quant="q4_k"))
