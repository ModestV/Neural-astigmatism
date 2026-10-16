import math

import numpy as np
import pytest

from symspot.core.geometry_utils import (
    bbox_iou,
    prim_bbox,
    prim_length,
    sample_primitive,
    sample_primitives,
    similarity_from_two_points,
    transform_prim,
    umeyama,
    undirected_angle_diff,
)
from symspot.core.types import Arc, Segment, Similarity, TextPrim

RNG = np.random.default_rng(0)


def rand_sim(mirror: bool) -> Similarity:
    return Similarity(
        float(RNG.uniform(0.5, 2.0)),
        float(RNG.uniform(-math.pi, math.pi)),
        float(RNG.uniform(-10, 10)),
        float(RNG.uniform(-10, 10)),
        mirror,
    )


@pytest.mark.parametrize("mirror", [False, True])
def test_similarity_inverse_and_compose(mirror):
    A = rand_sim(mirror)
    B = rand_sim(not mirror)
    p = RNG.normal(size=(20, 2))
    np.testing.assert_allclose(A.inverse().apply(A.apply(p)), p, atol=1e-9)
    np.testing.assert_allclose(A.compose(B).apply(p), A.apply(B.apply(p)), atol=1e-9)
    np.testing.assert_allclose(A.matrix()[:2, :2], A.linear())


def test_similarity_mirror_flips_y_before_rotation():
    T = Similarity(1.0, 0.0, 0.0, 0.0, True)
    np.testing.assert_allclose(T.apply(np.array([1.0, 2.0])), [1.0, -2.0])
    T = Similarity(2.0, math.pi / 2, 1.0, 0.0, False)
    np.testing.assert_allclose(T.apply(np.array([1.0, 0.0])), [1.0, 2.0], atol=1e-12)


def test_similarity_dict_roundtrip():
    T = rand_sim(True)
    assert Similarity.from_dict(T.to_dict()) == T


@pytest.mark.parametrize("mirror", [False, True])
def test_umeyama_recovers_transform(mirror):
    T = rand_sim(mirror)
    src = RNG.normal(size=(30, 2))
    dst = T.apply(src)
    est = umeyama(src, dst, allow_mirror=True)
    assert est.mirror == mirror
    np.testing.assert_allclose(est.apply(src), dst, atol=1e-9)


def test_umeyama_no_mirror_when_forbidden():
    T = Similarity(1.3, 0.4, 2.0, -1.0, True)
    src = RNG.normal(size=(30, 2))
    est = umeyama(src, T.apply(src), allow_mirror=False)
    assert not est.mirror


def test_umeyama_weights_and_errors():
    T = rand_sim(False)
    src = RNG.normal(size=(10, 2))
    dst = T.apply(src)
    dst_bad = dst.copy()
    dst_bad[0] += 100.0
    w = np.ones(10)
    w[0] = 0.0
    est = umeyama(src, dst_bad, weights=w)
    np.testing.assert_allclose(est.apply(src[1:]), dst[1:], atol=1e-9)
    with pytest.raises(ValueError):
        umeyama(np.zeros((5, 2)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        umeyama(src[:1], dst[:1])


@pytest.mark.parametrize("mirror", [False, True])
def test_similarity_from_two_points(mirror):
    T = rand_sim(mirror)
    a = RNG.normal(size=(2, 2))
    b = T.apply(a)
    est = similarity_from_two_points(a[0], a[1], b[0], b[1], mirror=mirror)
    assert est is not None
    p = RNG.normal(size=(5, 2))
    np.testing.assert_allclose(est.apply(p), T.apply(p), atol=1e-9)
    assert similarity_from_two_points(a[0], a[0], b[0], b[1]) is None


def test_sampling_segment_midpoints():
    s = Segment(0, (0.0, 0.0), (1.0, 0.0), 0)
    pts, tan = sample_primitive(s, 0.25)
    np.testing.assert_allclose(pts[:, 0], [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(tan, [[1.0, 0.0]] * 4)
    assert sample_primitive(Segment(1, (1.0, 1.0), (1.0, 1.0), 0), 0.1)[0].shape == (0, 2)
    with pytest.raises(ValueError):
        sample_primitive(s, 0.0)


def test_sampling_arc_on_circle_and_tangent():
    a = Arc(0, (1.0, 2.0), 3.0, 0.0, 2 * math.pi, 0)
    pts, tan = sample_primitive(a, 0.1)
    d = np.hypot(pts[:, 0] - 1.0, pts[:, 1] - 2.0)
    np.testing.assert_allclose(d, 3.0)
    radial = (pts - np.array([1.0, 2.0])) / 3.0
    np.testing.assert_allclose((radial * tan).sum(1), 0.0, atol=1e-12)
    assert len(pts) == math.ceil(2 * math.pi * 3.0 / 0.1)


def test_sample_primitives_index_and_text():
    prims = [
        Segment(0, (0.0, 0.0), (1.0, 0.0), 0),
        TextPrim(1, "S", (0.0, 0.0, 1.0, 1.0), 1),
        Arc(2, (0.0, 0.0), 1.0, 0.0, math.pi, 2),
    ]
    pts, tan, idx = sample_primitives(prims, 0.5)
    assert set(idx.tolist()) == {0, 2}
    assert len(pts) == len(tan) == len(idx)
    e = sample_primitives([], 0.5)
    assert e[0].shape == (0, 2) and e[2].shape == (0,)


@pytest.mark.parametrize("mirror", [False, True])
def test_transform_arc_matches_transformed_samples(mirror):
    a = Arc(0, (1.0, 0.5), 2.0, 0.3, 1.9, 0)
    T = rand_sim(mirror)
    ta = transform_prim(a, T)
    assert isinstance(ta, Arc)
    assert ta.t0 < ta.t1
    assert math.isclose(prim_length(ta), prim_length(a) * T.s)
    pts, _ = sample_primitive(a, 0.05)
    tpts = T.apply(pts)
    d = np.hypot(tpts[:, 0] - ta.center[0], tpts[:, 1] - ta.center[1])
    np.testing.assert_allclose(d, ta.r)
    ang = np.arctan2(tpts[:, 1] - ta.center[1], tpts[:, 0] - ta.center[0])
    rel = (ang - ta.t0) % (2 * math.pi)
    assert np.all(rel <= ta.t1 - ta.t0 + 1e-9)


def test_transform_segment_and_text():
    T = Similarity(2.0, math.pi / 2, 1.0, 1.0)
    s = transform_prim(Segment(5, (0.0, 0.0), (1.0, 0.0), 7), T)
    assert s.id == 5 and s.parent_id == 7
    np.testing.assert_allclose([s.p0, s.p1], [[1.0, 1.0], [1.0, 3.0]], atol=1e-12)
    t = transform_prim(TextPrim(1, "A", (0.0, 0.0, 2.0, 1.0), 1), T)
    np.testing.assert_allclose(t.bbox, (-1.0, 1.0, 1.0, 5.0), atol=1e-12)


def test_prim_bbox_partial_arc():
    a = Arc(0, (0.0, 0.0), 1.0, -0.1, math.pi / 2 + 0.1, 0)
    b = prim_bbox(a)
    assert math.isclose(b[2], 1.0) and math.isclose(b[3], 1.0)
    assert b[0] < 0 and b[1] < 0 and b[0] > -0.2


def test_bbox_iou_and_angles():
    assert bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert undirected_angle_diff(0.1, math.pi + 0.1) == pytest.approx(0.0, abs=1e-12)
    assert undirected_angle_diff(0.0, math.pi / 2) == pytest.approx(math.pi / 2)
