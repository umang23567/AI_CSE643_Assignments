"""Unit tests for the DSL primitives and the search. Run:  python test_dsl.py"""
import unittest
import numpy as np
import dsl
from search import solve, colormap_fit, apply_map

A = lambda x: np.array(x, dtype=np.int8)


class TestPrimitives(unittest.TestCase):
    def test_geometry(self):
        g = A([[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(dsl.rot90(g), A([[2, 4], [1, 3]])))
        self.assertTrue(np.array_equal(dsl.fliplr(g), A([[2, 1], [4, 3]])))
        self.assertTrue(np.array_equal(dsl.transpose(g), A([[1, 3], [2, 4]])))
        self.assertTrue(np.array_equal(dsl.antitranspose(g), A([[4, 2], [3, 1]])))
        self.assertTrue(np.array_equal(dsl.rot270(dsl.rot90(g)), g))

    def test_scaling_tiling_mirroring(self):
        g = A([[1, 2]])
        self.assertTrue(np.array_equal(dsl.up2(g), A([[1, 1, 2, 2], [1, 1, 2, 2]])))
        self.assertEqual(dsl.tile22(g).shape, (2, 4))
        self.assertTrue(np.array_equal(dsl.mirror_h(g), A([[1, 2, 2, 1]])))
        self.assertEqual(dsl.mirror_4(g).shape, (2, 4))
        self.assertIsNone(dsl.down2(A([[1, 2, 3]])))

    def test_crop_and_objects(self):
        g = A([[0, 0, 0, 0], [0, 5, 5, 0], [0, 5, 0, 0], [0, 0, 0, 7]])
        self.assertTrue(np.array_equal(dsl.crop_bbox(g), g[1:4, 1:4]))
        self.assertTrue(np.array_equal(dsl.crop_obj("largest")(g), A([[5, 5], [5, 0]])))
        self.assertTrue(np.array_equal(dsl.crop_obj("smallest")(g), A([[7]])))
        self.assertEqual(dsl.keep_obj("smallest")(g).sum(), 7)
        self.assertEqual(dsl.remove_singletons(g)[3, 3], 0)
        self.assertEqual(dsl.remove_singletons(g)[1, 1], 5)
        self.assertEqual(len(dsl.components(g)), 2)

    def test_diagonal_connectivity(self):
        g = A([[1, 0], [0, 1]])
        self.assertEqual(len(dsl.components(g, diag=True)), 1)
        self.assertEqual(len(dsl.components(g, diag=False)), 2)

    def test_fill_holes_only_fills_enclosed_cells(self):
        g = A([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])
        out = dsl.fill_holes(3)(g)
        self.assertEqual(out[1, 1], 3)
        self.assertEqual(out[0, 3], 0)               # outside stays background
        open_g = A([[1, 1, 1], [1, 0, 0], [1, 1, 1]])
        self.assertTrue(np.array_equal(dsl.fill_holes(3)(open_g), open_g))   # leaks to the border

    def test_gravity_connect_dedup(self):
        g = A([[1, 0], [0, 0], [0, 2]])
        self.assertTrue(np.array_equal(dsl.gravity("down")(g), A([[0, 0], [0, 0], [1, 2]])))
        self.assertTrue(np.array_equal(dsl.gravity("up")(A([[0], [0], [3]])), A([[3], [0], [0]])))
        self.assertTrue(np.array_equal(dsl.gravity("left")(A([[0, 0, 4]])), A([[4, 0, 0]])))
        self.assertTrue(np.array_equal(dsl.connect_h(A([[2, 0, 0, 2]])), A([[2, 2, 2, 2]])))
        self.assertTrue(np.array_equal(dsl.connect_h(A([[2, 0, 3]])), A([[2, 0, 3]])))     # different colours: no line
        self.assertTrue(np.array_equal(dsl.dedup_rows(A([[1], [1], [2]])), A([[1], [2]])))

    def test_halves_and_logic(self):
        g = A([[1, 0, 0, 1, 1], [0, 1, 7, 0, 1]])           # width 5: middle column is a separator
        self.assertTrue(np.array_equal(dsl.half(1, True)(g), A([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(dsl.half(1, False)(g), A([[1, 1], [0, 1]])))
        self.assertTrue(np.array_equal(dsl.logic(1, "and")(g), A([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(dsl.logic(1, "xor")(g), A([[0, 1], [0, 0]])))

    def test_symmetry_completion(self):
        g = A([[1, 0, 0, 0]])
        self.assertTrue(np.array_equal(dsl.sym_lr(g), A([[1, 0, 0, 1]])))

    def test_colour_ops(self):
        g = A([[1, 2], [2, 0]])
        self.assertTrue(np.array_equal(dsl.remove_color(2)(g), A([[1, 0], [0, 0]])))
        self.assertTrue(np.array_equal(dsl.keep_color(2)(g), A([[0, 2], [2, 0]])))
        self.assertTrue(np.array_equal(dsl.replace_color(1, 9)(g), A([[9, 2], [2, 0]])))
        self.assertTrue(np.array_equal(dsl.crop_color(2)(g), g[0:2, 0:2]))
        b = dsl.draw_border(5)(A([[0] * 3] * 3))
        self.assertEqual(b[1, 1], 0)
        self.assertEqual(b[0, 0], 5)


class TestSearch(unittest.TestCase):
    def test_colormap_fit(self):
        self.assertEqual(colormap_fit([A([[1, 2]])], [A([[3, 2]])]), {1: 3})
        self.assertIsNone(colormap_fit([A([[1, 1]])], [A([[3, 4]])]))          # inconsistent
        self.assertEqual(colormap_fit([A([[1]])], [A([[1]])]), {})
        self.assertTrue(np.array_equal(apply_map(A([[1, 2]]), {1: 3}), A([[3, 2]])))

    def _task(self, fn, shapes=((3, 3), (4, 3), (3, 4))):
        rng = np.random.RandomState(0)
        ins = [rng.randint(0, 4, s).astype(np.int8) for s in shapes]
        return {"train": [{"input": g.tolist(), "output": fn(g).tolist()} for g in ins[:2]],
                "test": [{"input": ins[2].tolist(), "output": fn(ins[2]).tolist()}]}

    def test_finds_a_two_step_program_with_every_strategy(self):
        task = self._task(lambda g: np.rot90(g[:, ::-1], 1))                # a single transpose-like op
        for strat in ("bfs", "greedy", "astar"):
            res = solve(task, strategy=strat, max_depth=3, time_limit=20)
            self.assertTrue(res.solutions, strat)
            truth = np.array(task["test"][0]["output"])
            self.assertTrue(any(np.array_equal(p, truth) for p in res.attempts(0)), strat)

    def test_bfs_returns_the_shortest_program(self):
        task = self._task(lambda g: np.rot90(g, 2))                          # rot180 = 1 step
        res = solve(task, "bfs", max_depth=3)
        self.assertEqual(len(res.solutions[0].program), 1)

    def test_pure_recolour_needs_no_primitive(self):
        task = self._task(lambda g: np.where(g == 1, 7, g))
        res = solve(task, "bfs", max_depth=2)
        self.assertEqual(res.solutions[0].program, [])
        self.assertEqual(res.solutions[0].mapping, {1: 7})

    def test_unsolvable_task_reports_failure(self):
        rng = np.random.RandomState(1)
        task = {"train": [{"input": rng.randint(0, 5, (3, 3)).tolist(), "output": rng.randint(0, 5, (3, 3)).tolist()}
                          for _ in range(2)],
                "test": [{"input": rng.randint(0, 5, (3, 3)).tolist(), "output": [[0]]}]}
        res = solve(task, "bfs", max_depth=2, time_limit=5)
        self.assertFalse(res.solutions)


if __name__ == "__main__":
    unittest.main(verbosity=1)
