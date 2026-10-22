"""Сквозные тесты: маленький набор -> augment.py -> проверка рамок -> visualize.py."""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import augment  # noqa: E402
import visualize  # noqa: E402
from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet, find_problems, load_sheets, save_coco  # noqa: E402

W, H = 200, 120


def make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Один лист 200x120: образец и три объекта (чёрные прямоугольники), разметка и картинка."""
    images = tmp_path / "src_images"
    images.mkdir(parents=True)
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    specs = [
        (LEGEND_ITEM, (10, 10, 30, 20)),
        (OBJECT, (60, 10, 30, 20)),
        (OBJECT, (95, 50, 30, 20)),  # лежит на границе плиток 100x60
        (OBJECT, (150, 90, 30, 20)),
    ]
    boxes = []
    for category, (x, y, w, h) in specs:
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=0)
        boxes.append(Box(category, (x, y, w, h), {"legend_id": "1", "rotation": 0.0}))
    image.save(images / "plan.png")
    annotations = tmp_path / "src.json"
    save_coco([Sheet("plan.png", W, H, boxes)], annotations)
    return annotations, images


def run_cli(tmp_path: Path, *extra: str) -> Path:
    annotations, images = make_dataset(tmp_path)
    out = tmp_path / "out"
    code = augment.main(["--annotations", str(annotations), "--images", str(images),
                         "--out", str(out), *extra])
    assert code == 0
    return out


def assert_marks_dark(out: Path):
    """Внутри каждой рамки (без края) картинка тёмная: рамки не уехали со знаков."""
    for sheet in load_sheets([out / "annotations.json"]):
        with Image.open(out / "images" / sheet.file_name) as image:
            image = image.convert("L")
            for box in sheet.boxes:
                x, y, w, h = box.bbox
                inner = image.crop((round(x + w / 4), round(y + h / 4),
                                    round(x + 3 * w / 4), round(y + 3 * h / 4)))
                assert inner.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) < 90, (
                    sheet.file_name, box)


def test_geometry_and_tiles_keep_boxes_on_marks(tmp_path):
    out = run_cli(tmp_path, "--rotate", "90", "270", "--scale", "0.5", "--mirror", "h",
                  "--tile", "100x60", "--overlap", "10", "--noise", "10", "--blur", "0.5",
                  "--stains", "5")
    sheets = load_sheets([out / "annotations.json"])
    assert len(sheets) > 10
    assert find_problems(sheets, out / "images") == []
    assert_marks_dark(out)
    names = {s.file_name for s in sheets}
    assert any("__rot90__tile_" in n for n in names) and any(n.endswith("__photo.png") for n in names)
    assert any("flip:h" in s.history for s in sheets)


def test_legend_link_survives(tmp_path):
    out = run_cli(tmp_path, "--rotate", "180", "--tile", "100x60")
    for sheet in load_sheets([out / "annotations.json"]):
        legend_ids = {b.legend_id for b in sheet.boxes if b.category == LEGEND_ITEM}
        for box in sheet.boxes:
            assert box.legend_id == "1"
            if box.category == OBJECT:
                assert box.attributes["legend_in_tile"] == (box.legend_id in legend_ids)


def test_require_mode_gives_only_tiles_with_legend(tmp_path):
    out = run_cli(tmp_path, "--tile", "100x60", "--legend-mode", "require")
    sheets = load_sheets([out / "annotations.json"])
    assert sheets
    for sheet in sheets:
        assert any(b.category == LEGEND_ITEM for b in sheet.boxes)
        assert all(b.attributes["legend_in_tile"] for b in sheet.boxes if b.category == OBJECT)


def test_no_clean_keeps_only_distorted_copies(tmp_path):
    out = run_cli(tmp_path, "--noise", "5", "--no-clean")
    sheets = load_sheets([out / "annotations.json"])
    assert [s.file_name for s in sheets] == ["plan__photo.png"]
    assert sorted(p.name for p in (out / "images").iterdir()) == ["plan__photo.png"]


def test_same_seed_gives_same_result(tmp_path):
    a = run_cli(tmp_path / "a", "--noise", "10", "--seed", "3")
    b = run_cli(tmp_path / "b", "--noise", "10", "--seed", "3")
    assert (a / "images" / "plan__photo.png").read_bytes() == (b / "images" / "plan__photo.png").read_bytes()


def test_missing_and_mismatched_images_are_skipped(tmp_path):
    annotations, images = make_dataset(tmp_path)
    (images / "plan.png").unlink()
    summary = augment.run(augment.build_parser().parse_args(
        ["--annotations", str(annotations), "--images", str(images), "--out", str(tmp_path / "o")]))
    assert summary["sheets"] == [] and "нет файла" in summary["skipped"][0]
    Image.new("L", (50, 50), 255).save(images / "plan.png")
    summary = augment.run(augment.build_parser().parse_args(
        ["--annotations", str(annotations), "--images", str(images), "--out", str(tmp_path / "o2")]))
    assert summary["sheets"] == [] and "не совпадает" in summary["skipped"][0]


@pytest.mark.parametrize("argv", [["--no-clean"], ["--no-original"]])
def test_useless_option_combinations_rejected(tmp_path, argv):
    annotations, images = make_dataset(tmp_path)
    with pytest.raises(SystemExit):
        augment.main(["--annotations", str(annotations), "--images", str(images),
                      "--out", str(tmp_path / "o"), *argv])


def test_tile_size_parser():
    assert augment.parse_tile("1600x900") == (1600, 900)
    with pytest.raises(Exception):
        augment.parse_tile("1600")


def test_visualize_draws_boxes_in_legend_color(tmp_path):
    out = run_cli(tmp_path, "--rotate", "90", "--tile", "100x60")
    preview = tmp_path / "preview"
    code = visualize.main([str(out / "annotations.json"), "--images", str(out / "images"),
                           "--out", str(preview), "--max", "3", "--max-side", "150"])
    assert code == 0
    files = sorted(preview.glob("preview_*.png"))
    assert len(files) == 3
    color = visualize.color_for("1")
    for file in files:
        with Image.open(file) as image:
            assert max(image.size) <= 150
            colors = {c for _, c in image.convert("RGB").getcolors(maxcolors=1_000_000)}
            assert color in colors


def test_visualize_reports_problems(tmp_path):
    annotations, images = make_dataset(tmp_path)
    (images / "plan.png").unlink()
    code = visualize.main([str(annotations), "--images", str(images), "--out", str(tmp_path / "p")])
    assert code == 1


def test_color_for_is_stable_and_gray_without_legend_id():
    assert visualize.color_for("1") == visualize.color_for("1")
    assert visualize.color_for("1") != visualize.color_for("2")
    assert visualize.color_for(None) == (128, 128, 128)
