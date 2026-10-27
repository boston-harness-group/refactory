"""The characters no path an agent names may hold (dm-208), checked in one place.

Materialise (`repo_change`), create mode (`create_ops`) and a workspace's collect step
(`workspace`) all take file names from an agent. They refuse the same names with the same
reason, so a builder that names a file in an emitted block and one that creates it in its
workspace are told the same thing (dm-278). It is core, beside `workspace`, which may not
import the product's `repo_change`; `repo_change` re-exports it.
"""

from __future__ import annotations

import unicodedata


#: The format characters a path may not hold (dm-208): the bidirectional controls, which
#: make a name read in another order than it is stored, and the invisible ones that make
#: two different names look the same. Other format characters stay allowed: U+200C and
#: U+200D join letters in Persian and other scripts, and characters in emoji sequences.
HIDDEN_FORMAT = frozenset(
    "\u061c\u200e\u200f" + "".join(map(chr, range(0x202A, 0x202F)))
    + "".join(map(chr, range(0x2066, 0x206A))) + "\u200b\u2060\ufeff")


def _unprintable(path: str) -> str:
    """The first character of ``path`` that a file name may not hold, named; else ``""``.

    Control characters (Unicode category Cc: the C0 range, DEL and the C1 range) and the
    format characters in `HIDDEN_FORMAT` (bidirectional controls such as U+202E, zero-width
    space, word joiner, byte-order mark), which a terminal, a diff or a review hides or
    shows as something else (dm-208). And a lone surrogate (category Cs): a file name
    that is not UTF-8, which git and Python decode to U+DC80..U+DCFF, and which no emission
    can be written with (dm-278).
    """
    for char in path:
        if unicodedata.category(char) == "Cs":
            byte = f"the byte 0x{ord(char) - 0xDC00:02X}" if 0xDC80 <= ord(char) <= 0xDCFF else "a lone surrogate"
            return f"U+{ord(char):04X} ({byte}, which is not UTF-8)"
        if unicodedata.category(char) == "Cc" or char in HIDDEN_FORMAT:
            name = unicodedata.name(char, "")
            return f"U+{ord(char):04X} " + (name if name else "(a control character)")
    return ""


def unprintable_refusal(path: str) -> str:
    """Why ``path`` is refused when it holds a character `_unprintable` names; else ``""``.

    The one check for emitted paths: materialise's `repo_change._artifact_paths`, create
    mode's `create_ops._package_relative` and a workspace's `workspace.collect` use it, and
    so should any other place that takes a path from an agent.
    """
    found = _unprintable(path)
    return (f"the emitted path {path!r} holds {found}, and no emitted path may hold a control "
            "character, a hidden format character (DEL, a C1 control, a bidirectional "
            "control, a zero-width space) or a byte that is not UTF-8: it is hidden or misshown wherever the path is read. "
            "Name the file without it") if found else ""
