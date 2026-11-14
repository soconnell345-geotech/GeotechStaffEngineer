"""Throwaway proof for INF-81 (branch ci/annot-proof only, never merged): two
tests that fail and one file-level error's stand-in, with made-up values, so
that the step that names failed tests can be read from the job's annotations."""
import pytest


def test_made_up_value_differs():
    assert 1.25 == pytest.approx(1.5, rel=0.01)


@pytest.mark.parametrize("case", ["50% of made-up", "second line"])
def test_made_up_raises(case):
    raise ValueError("no root for %s" % case)


def test_made_up_passes():
    assert 2 + 2 == 4
