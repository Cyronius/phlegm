"""Model recipes: compose the open kernels per model family from a ModelSpec.

Phase 3 of the open-kernels plan (.claude/plans/open-kernels-phase3-model-recipes.md):
the model's hyperparameters live in ONE place (`spec.ModelSpec`), every byte
layout, dispatch geometry, pool-packing law and per-layer program the driver
needs is derived from it by the family's recipe (`qwen36moe.py`), and the
driver reads the result from `manifest.json` (`manifest.py`) instead of
mirroring constants by hand.

    spec.py       ModelSpec: the hyperparameter tuple, from an HF config.json or GGUF metadata
    catalogue.py  the kernel templates and the parameter points they are validated for
    qwen36moe.py  the Qwen3.5/3.6-MoE recipe: layouts, designs, packing plan, step program
    pack.py       interpreter of a packing plan over a .q4nx container (NumPy)
    manifest.py   manifest.json: everything src/open_qwen36 reads
    cache.py      the build key (recipe + kernel sources + spec + quant)
    load.py       which spec a design build uses (OPEN_KERNELS_SPEC, else the checked-in 27B)
"""
