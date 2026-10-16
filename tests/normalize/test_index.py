import numpy as np

from symspot.core.geometry_utils import sample_primitives
from symspot.core.types import Arc, Segment, TextPrim
from symspot.normalize.index import PrimIndex


def _prims():
    return [
        Segment(10, (0, 0), (10, 0), 0),
        TextPrim(11, "A1", (4, 4, 6, 6), 1),
        Arc(12, (20, 20), 5, 0.0, 3.0, 2),
        Segment(13, (50, 50), (60, 50), 3),
        TextPrim(14, "B", (100, 100, 110, 110), 4),
    ]


def test_basic_attributes_and_sample_start():
    prims = _prims()
    idx = PrimIndex(prims, 1.0)
    assert list(idx.ids) == [10, 11, 12, 13, 14]
    assert idx.id_to_idx[13] == 3
    assert idx.bboxes.shape == (5, 4)
    pts, tan, sp = sample_primitives(prims, 1.0)
    np.testing.assert_allclose(idx.samples, pts)
    assert idx.sample_start.shape == (6,)
    for i in range(len(prims)):
        sl = slice(idx.sample_start[i], idx.sample_start[i + 1])
        assert np.all(idx.sample_prim[sl] == i)
    assert idx.sample_start[2] == idx.sample_start[1]  # у текста нет сэмплов
    assert idx.sample_start[-1] == len(idx.samples)
    assert idx.kdtree is not None and idx.sample_step == 1.0


def test_query_bbox_sorted():
    idx = PrimIndex(_prims(), 1.0)
    got = idx.query_bbox((3, -1, 30, 25))
    assert list(got) == [0, 1, 2]
    assert len(idx.query_bbox((200, 200, 300, 300))) == 0
    assert list(idx.query_bbox((55, 49, 56, 51))) == [3]


def test_texts_in_bbox_center_rule():
    idx = PrimIndex(_prims(), 1.0)
    assert [t.text for t in idx.texts_in_bbox((0, 0, 7, 7))] == ["A1"]
    assert idx.texts_in_bbox((0, 0, 4.5, 4.5)) == []  # центр (5, 5) снаружи
    assert [t.id for t in idx.texts_in_bbox((-1000, -1000, 1000, 1000))] == [11, 14]


def test_nearest_sample_and_radius():
    idx = PrimIndex(_prims(), 1.0)
    pts = np.array([[0.4, 0.1], [500.0, 500.0], [55.0, 50.5]])
    dist, ind = idx.nearest_sample(pts, 2.0)
    assert ind[1] == -1 and np.isinf(dist[1])
    assert ind[0] >= 0 and dist[0] < 0.2
    assert idx.sample_prim[ind[2]] == 3
    near = idx.samples_in_radius(pts, 1.0)
    assert len(near) == 3 and len(near[1]) == 0 and len(near[0]) > 0
    for k in near[2]:
        assert np.hypot(*(idx.samples[k] - pts[2])) <= 1.0 + 1e-9


def test_nearest_sample_max_dist_respected():
    idx = PrimIndex([Segment(0, (0, 0), (1, 0), 0)], 1.0)  # один сэмпл в (0.5, 0)
    dist, ind = idx.nearest_sample(np.array([[0.5, 3.0]]), 2.0)
    assert np.isinf(dist[0]) and ind[0] == -1
    dist, ind = idx.nearest_sample(np.array([[0.5, 1.0]]), 2.0)
    assert dist[0] == 1.0 and ind[0] == 0


def test_empty_index():
    idx = PrimIndex([], 1.0)
    assert idx.kdtree is None
    assert len(idx.query_bbox((0, 0, 1, 1))) == 0
    assert idx.texts_in_bbox((0, 0, 1, 1)) == []
    assert idx.sample_start.tolist() == [0]
    dist, ind = idx.nearest_sample(np.array([[0.0, 0.0], [1.0, 1.0]]), 5.0)
    assert np.all(np.isinf(dist)) and np.all(ind == -1)
    assert [len(x) for x in idx.samples_in_radius(np.zeros((2, 2)), 1.0)] == [0, 0]


def test_only_text_has_no_kdtree():
    idx = PrimIndex([TextPrim(0, "x", (0, 0, 1, 1), 0)], 1.0)
    assert idx.kdtree is None
    assert list(idx.query_bbox((0, 0, 2, 2))) == [0]
    dist, ind = idx.nearest_sample(np.zeros((1, 2)), 1.0)
    assert np.isinf(dist[0]) and ind[0] == -1
