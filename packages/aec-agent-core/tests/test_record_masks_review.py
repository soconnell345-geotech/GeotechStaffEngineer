"""The review of the record's masker (``module_work/record_masks/REVIEW.md``):
one test per finding, each failing before its fix.

Every value here is made up (``Made0Up...``, ``example.test``), never a
live key of any vendor.
"""

from __future__ import annotations

import json
import time

import pytest

from aec_agent_core.error_text import (error_line, mask_error_result,
                                       mask_error_secrets)

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"


# ---------------------------------------------------------------------------
# a private-key block with no END line
# ---------------------------------------------------------------------------

def test_many_unfinished_key_blocks_do_not_hold_the_turn():
    # 8,000 BEGIN lines and no END: about 8 s before the fix (each BEGIN
    # searched to the end of the text), 268 s for a 2 MB text.
    text = "-----BEGIN PRIVATE KEY-----" * 8000
    t0 = time.perf_counter()
    out = mask_error_secrets(text)
    assert time.perf_counter() - t0 < 1.0
    assert out == "…"


def test_a_key_block_cut_short_is_masked_to_the_end():
    text = ("bad key\n-----BEGIN RSA PRIVATE KEY-----\nMADEUPMADEUPMADEUP\n"
            "MOREMADEUP")
    assert mask_error_secrets(text) == "bad key\n…"
    whole = ("bad key\n-----BEGIN PRIVATE KEY-----\nMADEUPMADEUP\n"
             "-----END PRIVATE KEY-----\nend")
    assert mask_error_secrets(whole) == "bad key\n…\nend"
