"""Read compiler dependency tokens without losing Windows path separators."""
from pathlib import Path
import re
import shlex


def dependencies(text: str) -> set[str]:
    text = text.replace('\\\n', '')
    # Make escapes whitespace and punctuation. A Windows separator before a
    # letter is a literal backslash, whereas shlex would silently discard it.
    text = re.sub(r'\\(?![\s\\#:$])', r'\\\\', text)
    return {str(Path(token)) for token in shlex.split(text)}
