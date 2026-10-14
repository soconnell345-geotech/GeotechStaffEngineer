"""wave_equation calculation tables (backlog GM-39): the bearing graph table
is the result's own arrays."""

import math

from wave_equation import (Cushion, discretize_pile, generate_bearing_graph,
                           get_hammer)


def test_bearing_graph_table_is_the_result_arrays():
    hammer = get_hammer("Vulcan 010")
    cushion = Cushion(stiffness=500000, cor=0.80)
    d, t = 0.3239, 0.009525
    area = math.pi / 4 * (d ** 2 - (d - 2 * t) ** 2)
    pile = discretize_pile(15.0, area, 200e6, segment_length=1.0)
    bg = generate_bearing_graph(hammer, cushion, pile, R_min=200, R_max=800,
                                R_step=200)
    (table,) = bg.get_tables()
    assert table.n_rows == len(bg.R_values) == 4
    assert table.column("Ultimate resistance R_ult") == list(bg.R_values)
    assert table.column("Blow count") == list(bg.blow_counts)
    assert table.column("Permanent set s") == list(bg.permanent_sets)
    assert table.column("Maximum tension stress") == list(
        bg.max_tens_stresses)
