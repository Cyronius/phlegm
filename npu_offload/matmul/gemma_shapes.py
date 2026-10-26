"""The twelve Embedding-Gemma projection shapes shipped by FLM."""

SHAPES = [(m, k, n) for m in (512, 2048) for k, n in
          ((768, 768), (768, 1152), (768, 3072), (768, 256),
           (1152, 768), (3072, 768))]


def tag(shape: tuple[int, int, int]) -> str:
    m, k, n = shape
    return f"m{m}_{k}x{n}"
