"""The code voter's reading of the vision answer's READ and ANSWER lines
(2026-11-10 review, item 6): in a field that restates the quantity the
value is what follows the last equals sign. A recorded read of
"x = s_u2/s_u1 = 0.30" was measured at 2.0, the digit in the symbol's
name. Made-up values.
"""

from funhouse_agent.chart_reading import parse_reads


def test_the_value_follows_the_last_equals_sign():
    reads, _ = parse_reads(
        "About 4.1.\n"
        "READ | x = s_u2/s_u1 = 0.30 | curve = D/B = 0.5 | value = 4.1 | "
        "px=[10, 20, 30, 40]")
    assert len(reads) == 1
    assert reads[0]["axis"] == "x"
    assert reads[0]["at"] == 0.30
    assert reads[0]["curve"] == "0.5"
    assert reads[0]["value"] == 4.1
    assert reads[0]["px"] == [10.0, 20.0, 30.0, 40.0]


def test_a_plain_field_reads_as_before():
    reads, _ = parse_reads("READ | y = 12 | curve = 0.4 | value = 0.25 | "
                           "px=[1, 2, 3, 4]")
    assert reads[0]["axis"] == "y" and reads[0]["at"] == 12.0
    assert reads[0]["curve"] == "0.4" and reads[0]["value"] == 0.25


def test_the_answer_line_too():
    _, answer = parse_reads("ANSWER | curve = H/B = 0.75 | value = Nc = 6.2")
    assert answer == {"curve": 0.75, "value": 6.2}


def test_a_colon_field():
    reads, _ = parse_reads("READ | x: phi: 32 | curve: - | value: 18 | "
                           "px=[5, 6, 7, 8]")
    assert reads[0]["at"] == 32.0
    assert reads[0]["curve"] is None
    assert reads[0]["value"] == 18.0
