"""Streamlit-демо SymSpot: загрузка чертежа и эталонов, поиск, оверлей, таблица, метрики.

Запуск: ``uv run streamlit run app/demo.py``.

Логика, не зависящая от Streamlit, вынесена в функции модуля (``save_upload``, ``run_search``,
``build_table``, ``compute_metrics``, ``build_downloads`` и др.) и тестируется отдельно.
"""

from __future__ import annotations

import html
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from symspot.core.config import Config, load_config
from symspot.core.types import Drawing, Result
from symspot.render._common import instance_ids
from symspot.render.palette import color_hex

# --- константы интерфейса и вывода (не пороги алгоритма) ---
DRAWING_TYPES = ["pdf", "svg", "png", "jpg", "jpeg"]
QUERY_TYPES = ["json"]
OVERLAY_TARGET_WIDTH = 2000  # ширина PNG-оверлея, px
OVERLAY_MAX_WIDTH = 2400  # шире этого растровый оверлей уменьшается для показа
METRICS_IOU = 0.5
EXAMPLE_SIZE = "small"
EXAMPLE_NAME = "small_000"
HTML_VIEW_HEIGHT = 800
TABLE_COLUMNS = [
    "id",
    "legend_id",
    "label",
    "score",
    "coverage",
    "status",
    "source",
    "is_query",
    "rivals",
]
AMBIGUOUS_BG = "#ffe9a8"
DOWNLOAD_FILES = {
    "svg": ("overlay.svg", "image/svg+xml"),
    "pdf": ("overlay.pdf", "application/pdf"),
    "json": ("result.json", "application/json"),
    "html": ("report.html", "text/html"),
}
# Границы и шаги слайдеров; значения по умолчанию берутся из Config().
SLIDERS = {
    "accept_score": (0.3, 1.0, 0.01),
    "cover_tol": (0.005, 0.1, 0.005),
    "scale_min": (0.3, 1.0, 0.05),
    "scale_max": (1.0, 3.0, 0.05),
}


# ---------------------------------------------------------------------------
# Логика без Streamlit
# ---------------------------------------------------------------------------


@dataclass
class RunOutput:
    """Всё, что нужно для отображения и скачивания; кладется в ``st.session_state``."""

    result: Result
    drawing: Drawing
    table: pd.DataFrame
    metrics: dict | None
    metrics_note: str
    png: bytes
    downloads: dict[str, bytes | None]
    wall_seconds: float
    download_errors: dict[str, str] = field(default_factory=dict)
    html_report: str = ""


def new_workdir(prefix: str = "symspot_demo_") -> Path:
    """Временный каталог для загрузок и результатов."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def save_upload(name: str, data: bytes, dest_dir: str | Path) -> Path:
    """Сохранить загруженный файл под исходным именем (без путей) и вернуть путь."""
    safe = Path(name).name
    if not safe:
        raise ValueError("пустое имя файла")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / safe
    path.write_bytes(data)
    return path


def make_example(seed: int, dest_dir: str | Path) -> tuple[Path, Path]:
    """Сгенерировать один small-чертеж (SVG, без растра); вернуть (svg, gt json)."""
    from symspot.synth.export import generate

    dest = Path(dest_dir)
    names = generate(1, EXAMPLE_SIZE, int(seed), dest, raster=False)
    return dest / f"{names[0]}.svg", dest / f"{names[0]}_gt.json"


def build_config(
    accept_score: float | None = None,
    cover_tol: float | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    allow_mirror: bool | None = None,
) -> Config:
    """Конфиг по умолчанию с переопределением порогов из интерфейса."""
    overrides = {
        "accept_score": accept_score,
        "cover_tol": cover_tol,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "allow_mirror": allow_mirror,
    }
    return load_config(None, overrides)


def label_map(result: Result) -> dict[str, str]:
    """legend_id -> подпись эталона (если есть)."""
    out: dict[str, str] = {}
    for q in result.queries:
        if q.label and q.legend_id not in out:
            out[q.legend_id] = q.label
    return out


def build_table(result: Result) -> pd.DataFrame:
    """Таблица экземпляров; ``id`` совпадает с номерами в SVG/HTML-оверлее."""
    labels = label_map(result)
    ids = instance_ids(result)
    rows = [
        {
            "id": eid,
            "legend_id": inst.legend_id,
            "label": labels.get(inst.legend_id, ""),
            "score": round(inst.score, 3),
            "coverage": round(inst.coverage, 3),
            "status": inst.status,
            "source": inst.source,
            "is_query": inst.is_query,
            "rivals": ", ".join(inst.rivals),
        }
        for eid, inst in zip(ids, result.instances, strict=True)
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def filter_table(
    table: pd.DataFrame, classes: list[str] | None = None, show_queries: bool = True
) -> pd.DataFrame:
    """Фильтр по классам (пустой список или None = все) и по признаку эталона."""
    out = table
    if classes:
        out = out[out["legend_id"].isin(classes)]
    if not show_queries:
        out = out[~out["is_query"]]
    return out.reset_index(drop=True)


def ambiguous_table(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"] == "ambiguous"].reset_index(drop=True)


def class_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Число найденных экземпляров по классам (без эталонов) и из них ambiguous."""
    found = table[~table["is_query"]]
    if found.empty:
        return pd.DataFrame(columns=["legend_id", "found", "ambiguous"])
    g = found.groupby("legend_id")["status"]
    return pd.DataFrame(
        {"found": g.size(), "ambiguous": g.apply(lambda s: int((s == "ambiguous").sum()))}
    ).reset_index()


def compute_metrics(result: Result, gt_path: str | Path, iou: float = METRICS_IOU) -> dict:
    """Метрики относительно разметки (GT синтетики или COCO с ``object``)."""
    from symspot.cli import compute_metrics

    return compute_metrics(result, Path(gt_path), iou)


def try_metrics(result: Result, gt_path: str | Path | None) -> tuple[dict | None, str]:
    """Метрики, если в файле есть разметка экземпляров; иначе (None, причина)."""
    if gt_path is None:
        return None, "Файл эталонов не задан."
    try:
        return compute_metrics(result, gt_path), ""
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        return None, f"В файле нет разметки экземпляров для метрик ({e})."


def metrics_table(metrics: dict) -> pd.DataFrame:
    """P/R/F1 и tp/fp/fn по классам, последняя строка ИТОГО."""
    rows = []
    for cls, v in metrics["per_class"].items():
        rows.append({"класс": cls, **{k: v[k] for k in ("precision", "recall", "f1")}})
        rows[-1].update({k: v[k] for k in ("tp", "fp", "fn")})
    o = metrics["overall"]
    rows.append({"класс": "ИТОГО", **{k: o[k] for k in ("precision", "recall", "f1")}})
    rows[-1].update({k: o[k] for k in ("tp", "fp", "fn")})
    df = pd.DataFrame(rows)
    for col in ("precision", "recall", "f1"):
        df[col] = df[col].round(3)
    return df


def overlay_png_bytes(
    drawing: Drawing, result: Result, work_dir: str | Path, target_width: int = OVERLAY_TARGET_WIDTH
) -> bytes:
    """PNG-оверлей: вектор растеризуется с масштабом под ширину ``target_width``,
    крупный растр уменьшается до ``OVERLAY_MAX_WIDTH``."""
    from symspot.render.png_overlay import render_png

    out = Path(work_dir) / "overlay.png"
    width = max(float(result.width), 1.0)
    path = render_png(drawing, result, out, scale=target_width / width)
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None and img.shape[1] > OVERLAY_MAX_WIDTH:
        k = OVERLAY_TARGET_WIDTH / img.shape[1]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        if ok:
            return buf.tobytes()
    return data


def build_downloads(
    drawing: Drawing, result: Result, work_dir: str | Path
) -> tuple[dict[str, bytes | None], dict[str, str]]:
    """Байты файлов для скачивания: svg, pdf, json, html. Ошибка рендера одного формата
    не ломает остальные: для него None и текст ошибки во втором словаре."""
    from symspot.render.html_report import render_html
    from symspot.render.pdf_overlay import render_pdf
    from symspot.render.svg_overlay import render_svg

    out = Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderers = {"svg": render_svg, "pdf": render_pdf, "html": render_html}
    data: dict[str, bytes | None] = {}
    errors: dict[str, str] = {}
    for key, fn in renderers.items():
        try:
            data[key] = Path(fn(drawing, result, out / DOWNLOAD_FILES[key][0])).read_bytes()
        except Exception as e:  # noqa: BLE001 - один формат не должен ломать остальные
            data[key] = None
            errors[key] = f"{type(e).__name__}: {e}"
    data["json"] = json.dumps(result.to_dict(), ensure_ascii=False, indent=1).encode("utf-8")
    return data, errors


def run_search(
    drawing_path: str | Path,
    queries_path: str | Path,
    cfg: Config,
    page: int = 0,
    work_dir: str | Path | None = None,
    gt_path: str | Path | None = None,
) -> RunOutput:
    """Полный прогон: поиск, таблица, метрики (по ``gt_path`` или по файлу эталонов),
    PNG-оверлей и байты скачивания."""
    from symspot.pipeline import run

    work = Path(work_dir) if work_dir is not None else new_workdir()
    t0 = time.perf_counter()
    result, norm = run(drawing_path, queries_path, cfg, page=page)
    metrics, note = try_metrics(result, gt_path if gt_path is not None else queries_path)
    downloads, errors = build_downloads(norm, result, work)
    png = overlay_png_bytes(norm, result, work)
    return RunOutput(
        result=result,
        drawing=norm,
        table=build_table(result),
        metrics=metrics,
        metrics_note=note,
        png=png,
        downloads=downloads,
        wall_seconds=time.perf_counter() - t0,
        download_errors=errors,
        html_report=(downloads["html"] or b"").decode("utf-8"),
    )


def legend_html(classes: list[str]) -> str:
    """Легенда цветов классов (одинаковых в оверлее, SVG, PDF и HTML)."""
    items = [
        f'<span style="display:inline-block;margin-right:14px">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{color_hex(c)};'
        f'margin-right:4px;vertical-align:middle"></span>{html.escape(c)}</span>'
        for c in classes
    ]
    return "".join(items)


# ---------------------------------------------------------------------------
# Интерфейс Streamlit
# ---------------------------------------------------------------------------


def _uploads_signature(drawing_file: Any, queries_file: Any) -> tuple:
    return tuple((f.name, f.size) if f is not None else None for f in (drawing_file, queries_file))


def _sidebar(st: Any) -> dict[str, Any]:
    default = Config()
    ss = st.session_state
    st.sidebar.header("Входные данные")
    drawing_file = st.sidebar.file_uploader("Чертеж", type=DRAWING_TYPES, key="drawing_file")
    queries_file = st.sidebar.file_uploader(
        "Эталоны (COCO или простой JSON)", type=QUERY_TYPES, key="queries_file"
    )
    page = st.sidebar.number_input("Страница PDF", min_value=0, value=0, step=1, key="page")

    st.sidebar.header("Пороги")
    cfg_values: dict[str, Any] = {}
    for name, (lo, hi, step) in SLIDERS.items():
        cfg_values[name] = st.sidebar.slider(
            name, float(lo), float(hi), float(getattr(default, name)), float(step), key=name
        )
    cfg_values["allow_mirror"] = st.sidebar.checkbox(
        "allow_mirror", value=default.allow_mirror, key="allow_mirror"
    )

    st.sidebar.header("Пример")
    seed = st.sidebar.number_input("Seed синтетики", min_value=0, value=0, step=1, key="seed")
    if st.sidebar.button("Пример", key="btn_example"):
        if "workdir" not in ss:
            ss["workdir"] = str(new_workdir())
        svg, gt = make_example(int(seed), Path(ss["workdir"]) / "example")
        ss["example"] = {
            "drawing": str(svg),
            "queries": str(gt),
            "signature": _uploads_signature(drawing_file, queries_file),
        }
    st.sidebar.divider()
    go = st.sidebar.button("Найти", type="primary", key="btn_run")
    return {
        "drawing_file": drawing_file,
        "queries_file": queries_file,
        "page": int(page),
        "cfg": cfg_values,
        "go": go,
    }


def _resolve_inputs(st: Any, ui: dict[str, Any]) -> tuple[Path, Path] | None:
    """Выбрать входные файлы: свежая загрузка важнее примера. Загрузки сохраняются
    во временный каталог под исходными именами."""
    ss = st.session_state
    ex = ss.get("example")
    sig = _uploads_signature(ui["drawing_file"], ui["queries_file"])
    if ex is not None and sig == ex["signature"]:
        return Path(ex["drawing"]), Path(ex["queries"])
    if ui["drawing_file"] is None or ui["queries_file"] is None:
        return None
    if "workdir" not in ss:
        ss["workdir"] = str(new_workdir())
    up_dir = Path(ss["workdir"]) / "upload"
    d = save_upload(ui["drawing_file"].name, ui["drawing_file"].getvalue(), up_dir)
    q = save_upload(ui["queries_file"].name, ui["queries_file"].getvalue(), up_dir)
    return d, q


def _show_results(st: Any, out: RunOutput) -> None:
    result = out.result
    found = out.table[~out.table["is_query"]]
    n_amb = int((found["status"] == "ambiguous").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Найдено", len(found))
    c2.metric("Эталонов среди них", int(out.table["is_query"].sum()))
    c3.metric("Ambiguous", n_amb)
    c4.metric("Время, с", f"{result.timings.get('total', out.wall_seconds):.2f}")

    classes = sorted(out.table["legend_id"].unique())
    st.markdown(legend_html(classes), unsafe_allow_html=True)

    view = st.radio(
        "Вид", ["Оверлей (PNG)", "Интерактивный (zoom/pan)"], horizontal=True, key="view"
    )
    if view.startswith("Оверлей"):
        st.image(out.png, width="stretch")
    elif out.html_report:
        st.iframe(out.html_report, height=HTML_VIEW_HEIGHT)
    else:
        st.warning("HTML-отчет недоступен: " + out.download_errors.get("html", "нет данных"))

    st.subheader("Скачать")
    cols = st.columns(len(DOWNLOAD_FILES))
    for col, (key, (fname, mime)) in zip(cols, DOWNLOAD_FILES.items(), strict=True):
        data = out.downloads.get(key)
        col.download_button(
            fname, data or b"", file_name=fname, mime=mime, disabled=not data, key=f"dl_{key}"
        )
        if key in out.download_errors:
            col.caption(out.download_errors[key])

    st.subheader("Экземпляры")
    chosen = st.multiselect("Фильтр по классу", classes, key="class_filter")
    show_q = st.checkbox("Показывать эталоны", value=True, key="show_queries")
    shown = filter_table(out.table, chosen, show_q)
    styled = shown.style.apply(
        lambda row: (
            [f"background-color: {AMBIGUOUS_BG}" if row["status"] == "ambiguous" else ""] * len(row)
        ),
        axis=1,
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    amb = ambiguous_table(filter_table(out.table, chosen, show_q))
    st.subheader(f"Ambiguous ({len(amb)})")
    if amb.empty:
        st.caption("Неоднозначных экземпляров нет.")
    else:
        st.dataframe(amb, width="stretch", hide_index=True)

    st.subheader("По классам")
    st.dataframe(class_summary(out.table), width="stretch", height="content", hide_index=True)

    st.subheader("Метрики")
    if out.metrics is None:
        st.caption(out.metrics_note)
    else:
        st.dataframe(metrics_table(out.metrics), width="stretch", height="content", hide_index=True)
        st.caption(
            f"IoU >= {METRICS_IOU}; similar_group_error = "
            f"{out.metrics['similar_group_error']:.3f} (символов в группах: "
            f"{out.metrics['n_similar']})"
        )

    st.subheader("Тайминги")
    st.dataframe(
        pd.DataFrame(
            {
                "этап": list(result.timings),
                "секунды": [round(v, 3) for v in result.timings.values()],
            }
        ),
        width="stretch",
        height="content",
        hide_index=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SymSpot", layout="wide")
    st.title("SymSpot: поиск обозначений на чертеже")
    ui = _sidebar(st)
    ss = st.session_state

    if ui["go"]:
        inputs = _resolve_inputs(st, ui)
        if inputs is None:
            st.error("Загрузите чертеж и файл эталонов или нажмите «Пример».")
        else:
            drawing_path, queries_path = inputs
            try:
                cfg = build_config(**ui["cfg"])
                work = Path(ss["workdir"]) if "workdir" in ss else new_workdir()
                ss["workdir"] = str(work)
                with st.spinner("Поиск обозначений..."):
                    ss["out"] = run_search(
                        drawing_path,
                        queries_path,
                        cfg,
                        page=ui["page"],
                        work_dir=work / "out",
                    )
                ss["out_name"] = drawing_path.name
            except Exception as e:  # noqa: BLE001 - показать пользователю, не падать
                ss.pop("out", None)
                st.error(f"Ошибка: {e}")

    out: RunOutput | None = ss.get("out")
    if out is None:
        st.info("Загрузите чертеж и эталоны в боковой панели или нажмите «Пример», затем «Найти».")
        if "example" in ss:
            st.caption(f"Пример готов: {Path(ss['example']['drawing']).name}")
        return
    st.caption(f"Чертеж: {ss.get('out_name', '')}")
    _show_results(st, out)


if __name__ == "__main__":
    main()
