import { useEffect, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router'
import { api, errorText } from '../api/client.ts'
import type { BBox, Drawing, SearchParams, SearchRun } from '../api/types.ts'
import {
  DrawingViewer,
  type Overlay,
  type Tool,
  type ViewerHandle,
} from '../components/DrawingViewer.tsx'
import { SearchParamsForm } from '../components/SearchParamsForm.tsx'
import { ErrorMessage, Loading, Notice, PageHeader, type NoticeKind } from '../components/ui.tsx'
import { useAsync } from '../hooks/useAsync.ts'
import { cropPage } from '../lib/crop.ts'
import { isTypingTarget } from '../lib/dom.ts'
import { FORMAT_LABELS, formatDate, formatNumber } from '../lib/format.ts'
import { clamp, roundBBox } from '../lib/geometry.ts'
import { DEFAULT_SEARCH_PARAMS } from '../lib/search.ts'

/**
 * Просмотр чертежа. Параметры адреса:
 *   ?page=N     — номер листа;
 *   ?run=ID     — показать результаты запуска поиска;
 *   ?symbol=ID  — показать образец из библиотеки.
 */
export function ViewerPage() {
  const { id = '' } = useParams()
  const drawing = useAsync(() => api.getDrawing(id), id)

  if (drawing.error) {
    return (
      <div className="page">
        <ErrorMessage error={drawing.error} onRetry={drawing.reload} />
        <Link to="/drawings">← К списку чертежей</Link>
      </div>
    )
  }
  if (!drawing.data) {
    return (
      <div className="page">
        <Loading />
      </div>
    )
  }
  if (drawing.data.status !== 'ready') {
    return <BrokenDrawing drawing={drawing.data} onReprocessed={drawing.reload} />
  }
  return <Viewer key={drawing.data.id} drawing={drawing.data} />
}

function Viewer({ drawing }: { drawing: Drawing }) {
  const [query, setQuery] = useSearchParams()
  const pageCount = drawing.pages.length
  const pageNumber = clamp(Math.trunc(Number(query.get('page'))) || 1, 1, pageCount)
  const page = drawing.pages[pageNumber - 1]
  const runId = query.get('run')
  const focusSymbolId = query.get('symbol')

  const library = useAsync(() => api.listSymbols())
  const viewerRef = useRef<ViewerHandle>(null)
  const pendingFocus = useRef<BBox | null>(null)

  const [tool, setTool] = useState<Tool>('select')
  const [selection, setSelection] = useState<{ page: number; bbox: BBox } | null>(null)
  const [params, setParams] = useState<SearchParams>(DEFAULT_SEARCH_PARAMS)
  const [allPages, setAllPages] = useState(false)
  const [checked, setChecked] = useState<Set<string>>(() => new Set())
  const [run, setRun] = useState<SearchRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<{ kind: NoticeKind; text: string } | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [showTemplates, setShowTemplates] = useState(true)
  const [saveForm, setSaveForm] = useState<{ name: string; category: string } | null>(null)
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null)

  // Выделение относится к конкретному листу и на других листах не показывается.
  const currentSelection = selection?.page === pageNumber ? selection.bbox : null
  // Подсвечиваем результат под курсором, а если такого нет — образец из адреса (?symbol=ID).
  const highlightedId = hovered ?? (focusSymbolId ? `sym-${focusSymbolId}` : null)
  const symbols = library.data ?? []
  const symbolNames = new Map(symbols.map((s) => [s.id, s.name]))
  const categories = [...new Set(symbols.flatMap((s) => (s.category ? [s.category] : [])))]

  function updateQuery(changes: Record<string, string | null>) {
    setQuery(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [key, value] of Object.entries(changes)) {
          if (value === null) next.delete(key)
          else next.set(key, value)
        }
        return next
      },
      { replace: true },
    )
  }

  function goToPage(number: number) {
    if (number >= 1 && number <= pageCount && number !== pageNumber) {
      updateQuery({ page: String(number) })
    }
  }

  /** Приблизить рамку; если она на другом листе — сначала перейти на него. */
  function focusOn(bbox: BBox, pageNo: number) {
    if (pageNo !== pageNumber) {
      pendingFocus.current = bbox
      goToPage(pageNo)
    } else if (loadedSrc === page.image_url) {
      viewerRef.current?.zoomTo(bbox)
    } else {
      pendingFocus.current = bbox
    }
  }

  // ?run=ID — подгружаем результаты запуска (например, перешли из «Поиска»).
  useEffect(() => {
    if (!runId || run?.id === runId) return
    let cancelled = false
    api.getRun(runId).then(
      (loaded) => {
        if (cancelled) return
        if (loaded.request.drawing_id === drawing.id) setRun(loaded)
        else setNotice({ kind: 'error', text: 'Этот запуск поиска относится к другому чертежу' })
      },
      (error) => {
        if (!cancelled) setNotice({ kind: 'error', text: `Запуск не загружен: ${errorText(error)}` })
      },
    )
    return () => {
      cancelled = true
    }
  }, [runId, run?.id, drawing.id])

  // ?symbol=ID — приближаем образец из библиотеки, как только она загрузится (один раз).
  const focusSymbol = symbols.find((s) => s.id === focusSymbolId && s.drawing_id === drawing.id)
  const focusedSymbolRef = useRef<string | null>(null)
  useEffect(() => {
    if (!focusSymbol || focusedSymbolRef.current === focusSymbol.id) return
    focusedSymbolRef.current = focusSymbol.id
    focusOn(focusSymbol.bbox, focusSymbol.page)
  })

  // Горячие клавиши. Подписка обновляется на каждом рендере, чтобы видеть свежее состояние.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target) || event.ctrlKey || event.metaKey || event.altKey) return
      const viewer = viewerRef.current
      switch (event.code) {
        case 'KeyH':
          setTool('pan')
          break
        case 'KeyR':
          setTool('select')
          break
        case 'Escape':
          setSelection(null)
          setSaveForm(null)
          break
        case 'Equal':
        case 'NumpadAdd':
          viewer?.zoomBy(1.25)
          break
        case 'Minus':
        case 'NumpadSubtract':
          viewer?.zoomBy(0.8)
          break
        case 'Digit0':
        case 'Numpad0':
          viewer?.fit()
          break
        case 'PageDown':
          goToPage(pageNumber + 1)
          break
        case 'PageUp':
          goToPage(pageNumber - 1)
          break
        default:
          return
      }
      event.preventDefault()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  async function search(kind: 'region' | 'symbols') {
    setBusy(true)
    setNotice(null)
    try {
      const result = await api.search({
        drawing_id: drawing.id,
        pages: allPages ? null : [pageNumber],
        query:
          kind === 'region' && currentSelection
            ? { drawing_id: drawing.id, page: pageNumber, bbox: roundBBox(currentSelection) }
            : null,
        symbol_ids: kind === 'symbols' ? [...checked] : [],
        params,
      })
      setRun(result)
      setHovered(null)
      updateQuery({ run: result.id })
    } catch (error) {
      setNotice({ kind: 'error', text: errorText(error) })
    } finally {
      setBusy(false)
    }
  }

  async function saveSymbol(form: { name: string; category: string }) {
    if (!currentSelection) return
    setBusy(true)
    setNotice(null)
    try {
      // Превью желательно, но не обязательно: если браузер не смог его нарисовать, сохраняем без него.
      const preview = await cropPage(page.image_url, page.width, page.height, currentSelection).catch(
        () => null,
      )
      const symbol = await api.createSymbol({
        name: form.name.trim(),
        category: form.category.trim() || null,
        drawing_id: drawing.id,
        page: pageNumber,
        bbox: roundBBox(currentSelection),
        preview_png: preview,
      })
      setSaveForm(null)
      setNotice({ kind: 'ok', text: `«${symbol.name}» добавлено в библиотеку` })
      library.reload()
    } catch (error) {
      setNotice({ kind: 'error', text: errorText(error) })
    } finally {
      setBusy(false)
    }
  }

  function toggleChecked(symbolId: string) {
    setChecked((prev) => {
      const next = new Set(prev)
      if (next.has(symbolId)) next.delete(symbolId)
      else next.add(symbolId)
      return next
    })
  }

  const overlays: Overlay[] = []
  if (showTemplates) {
    for (const s of symbols) {
      if (s.drawing_id === drawing.id && s.page === pageNumber) {
        overlays.push({ id: `sym-${s.id}`, bbox: s.bbox, kind: 'template', label: s.name })
      }
    }
  }
  run?.detections.forEach((detection, index) => {
    if (detection.page === pageNumber) {
      overlays.push({ id: `det-${index}`, bbox: detection.bbox, kind: 'match', label: String(index + 1) })
    }
  })

  return (
    <div className="viewer-page">
      <div className="toolbar">
        <Link to="/drawings" className="btn btn--small">
          ← Чертежи
        </Link>
        <strong className="toolbar__title" title={drawing.name}>
          {drawing.name}
        </strong>
        <span className="badge">{FORMAT_LABELS[drawing.format]}</span>

        {pageCount > 1 && (
          <div className="btn-group">
            <button
              type="button"
              className="btn btn--small"
              onClick={() => goToPage(pageNumber - 1)}
              disabled={pageNumber <= 1}
              title="Предыдущий лист (PageUp)"
            >
              ‹
            </button>
            <span className="toolbar__text">
              Лист {pageNumber} из {pageCount}
            </span>
            <button
              type="button"
              className="btn btn--small"
              onClick={() => goToPage(pageNumber + 1)}
              disabled={pageNumber >= pageCount}
              title="Следующий лист (PageDown)"
            >
              ›
            </button>
          </div>
        )}

        <div className="btn-group" role="group" aria-label="Инструмент">
          <button
            type="button"
            className={`btn btn--small${tool === 'pan' ? ' is-active' : ''}`}
            aria-pressed={tool === 'pan'}
            onClick={() => setTool('pan')}
            title="Перемещение (H). В любом режиме можно зажать пробел"
          >
            Рука
          </button>
          <button
            type="button"
            className={`btn btn--small${tool === 'select' ? ' is-active' : ''}`}
            aria-pressed={tool === 'select'}
            onClick={() => setTool('select')}
            title="Выделить обозначение рамкой (R)"
          >
            Рамка
          </button>
        </div>

        <div className="btn-group">
          <button
            type="button"
            className="btn btn--small"
            onClick={() => viewerRef.current?.zoomBy(0.8)}
            title="Уменьшить (−)"
          >
            −
          </button>
          <button
            type="button"
            className="btn btn--small"
            onClick={() => viewerRef.current?.zoomBy(1.25)}
            title="Увеличить (+)"
          >
            +
          </button>
          <button
            type="button"
            className="btn btn--small"
            onClick={() => viewerRef.current?.fit()}
            title="Показать лист целиком (0)"
          >
            Весь лист
          </button>
        </div>

        <label className="check">
          <input
            type="checkbox"
            checked={showTemplates}
            onChange={(event) => setShowTemplates(event.target.checked)}
          />
          Образцы на листе
        </label>
        <a className="btn btn--small toolbar__end" href={drawing.original_url}>
          Скачать оригинал
        </a>
      </div>

      <div className="viewer-page__body">
        <DrawingViewer
          ref={viewerRef}
          src={page.image_url}
          width={page.width}
          height={page.height}
          units={drawing.units}
          tool={tool}
          selection={currentSelection}
          onSelectionChange={(bbox) => {
            setSelection(bbox ? { page: pageNumber, bbox } : null)
            setSaveForm(null)
          }}
          overlays={overlays}
          highlightedId={highlightedId}
          onImageLoad={() => {
            setLoadedSrc(page.image_url)
            if (pendingFocus.current) {
              viewerRef.current?.zoomTo(pendingFocus.current)
              pendingFocus.current = null
            }
          }}
        />

        <aside className="sidebar">
          {notice && <Notice kind={notice.kind}>{notice.text}</Notice>}

          <section className="panel">
            <h2>Выделенный фрагмент</h2>
            {!currentSelection ? (
              <p className="muted">
                Выберите инструмент «Рамка» (R) и обведите на чертеже условное обозначение.
              </p>
            ) : (
              <>
                <SelectionPreview
                  src={page.image_url}
                  width={page.width}
                  height={page.height}
                  bbox={currentSelection}
                />
                <BBoxFields
                  bbox={currentSelection}
                  onChange={(bbox) => setSelection({ page: pageNumber, bbox })}
                />
                <div className="actions">
                  <button
                    type="button"
                    className="btn btn--primary"
                    disabled={busy}
                    onClick={() => void search('region')}
                  >
                    Найти похожие
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() => setSaveForm({ name: '', category: '' })}
                  >
                    В библиотеку
                  </button>
                  <button type="button" className="btn btn--ghost" onClick={() => setSelection(null)}>
                    Сбросить
                  </button>
                </div>
                {saveForm && (
                  <form
                    className="save-form"
                    onSubmit={(event) => {
                      event.preventDefault()
                      void saveSymbol(saveForm)
                    }}
                  >
                    <label className="field">
                      <span>Название</span>
                      <input
                        autoFocus
                        required
                        value={saveForm.name}
                        placeholder="например, Розетка"
                        onChange={(event) => setSaveForm({ ...saveForm, name: event.target.value })}
                      />
                    </label>
                    <label className="field">
                      <span>Категория</span>
                      <input
                        list="symbol-categories"
                        value={saveForm.category}
                        placeholder="например, Электрика"
                        onChange={(event) => setSaveForm({ ...saveForm, category: event.target.value })}
                      />
                      <datalist id="symbol-categories">
                        {categories.map((category) => (
                          <option key={category} value={category} />
                        ))}
                      </datalist>
                    </label>
                    <div className="actions">
                      <button
                        type="submit"
                        className="btn btn--primary"
                        disabled={busy || !saveForm.name.trim()}
                      >
                        Сохранить
                      </button>
                      <button type="button" className="btn btn--ghost" onClick={() => setSaveForm(null)}>
                        Отмена
                      </button>
                    </div>
                  </form>
                )}
              </>
            )}
          </section>

          {run && (
            <section className="panel">
              <h2>Результаты поиска</h2>
              <p className="muted small">
                {formatDate(run.created_at)} · алгоритм «{run.detector}» ·{' '}
                {formatNumber(run.duration_ms, 0)} мс
              </p>
              {run.message && (
                <Notice kind={run.status === 'failed' ? 'error' : 'warning'}>{run.message}</Notice>
              )}
              {run.detections.length === 0 ? (
                <p className="muted">Ничего не найдено.</p>
              ) : (
                <ol className="results">
                  {run.detections.map((detection, index) => (
                    <li key={index}>
                      <button
                        type="button"
                        className={`result${highlightedId === `det-${index}` ? ' is-highlighted' : ''}`}
                        onMouseEnter={() => setHovered(`det-${index}`)}
                        onMouseLeave={() => setHovered(null)}
                        onClick={() => focusOn(detection.bbox, detection.page)}
                      >
                        <span>
                          {index + 1}.{' '}
                          {detection.symbol_id
                            ? (symbolNames.get(detection.symbol_id) ?? 'обозначение')
                            : 'выделенный фрагмент'}
                        </span>
                        <span className="muted">
                          лист {detection.page} · {Math.round(detection.score * 100)}%
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
              <button
                type="button"
                className="btn btn--small btn--ghost"
                onClick={() => {
                  setRun(null)
                  updateQuery({ run: null })
                }}
              >
                Скрыть результаты
              </button>
            </section>
          )}

          <section className="panel">
            <h2>Параметры поиска</h2>
            <SearchParamsForm value={params} onChange={setParams} />
            <label className="check">
              <input
                type="checkbox"
                checked={allPages}
                disabled={pageCount === 1}
                onChange={(event) => setAllPages(event.target.checked)}
              />
              Искать на всех листах
            </label>
            <p className="hint">
              Сейчас подключена заглушка: параметры уходят на сервер, но на результат не влияют.
            </p>
          </section>

          <section className="panel">
            <h2>Образцы из библиотеки</h2>
            {library.error && <ErrorMessage error={library.error} onRetry={library.reload} />}
            {symbols.length === 0 ? (
              <p className="muted">
                Библиотека пуста. Выделите обозначение и нажмите «В библиотеку».
              </p>
            ) : (
              <>
                <ul className="checklist">
                  {symbols.map((s) => (
                    <li key={s.id}>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={checked.has(s.id)}
                          onChange={() => toggleChecked(s.id)}
                        />
                        {s.preview_url && <img className="thumb-xs" src={s.preview_url} alt="" />}
                        <span>{s.name}</span>
                      </label>
                    </li>
                  ))}
                </ul>
                <button
                  type="button"
                  className="btn"
                  disabled={busy || checked.size === 0}
                  onClick={() => void search('symbols')}
                >
                  Найти выбранные ({checked.size})
                </button>
              </>
            )}
          </section>

          <p className="hint">
            Колесо мыши — масштаб, пробел + перетаскивание — перемещение, Esc — снять выделение,
            PageUp / PageDown — листы.
          </p>
        </aside>
      </div>
    </div>
  )
}

function SelectionPreview(props: { src: string; width: number; height: number; bbox: BBox }) {
  const { src, width, height, bbox } = props
  const key = `${src}|${bbox.x}|${bbox.y}|${bbox.w}|${bbox.h}`
  const [result, setResult] = useState<{ key: string; url: string | null } | null>(null)

  useEffect(() => {
    let cancelled = false
    cropPage(src, width, height, bbox).then(
      (url) => {
        if (!cancelled) setResult({ key, url })
      },
      () => {
        if (!cancelled) setResult({ key, url: null })
      },
    )
    return () => {
      cancelled = true
    }
  }, [key, src, width, height, bbox])

  if (result?.key !== key) return <div className="preview preview--empty">…</div>
  if (!result.url) return <div className="preview preview--empty">Превью недоступно</div>
  return <img className="preview" src={result.url} alt="Выделенный фрагмент" />
}

/** Точная правка рамки. Значение применяется, когда поле теряет фокус или нажат Enter. */
function BBoxFields({ bbox, onChange }: { bbox: BBox; onChange: (bbox: BBox) => void }) {
  const fields: [keyof BBox, string][] = [
    ['x', 'x'],
    ['y', 'y'],
    ['w', 'ширина'],
    ['h', 'высота'],
  ]
  return (
    <div className="bbox-fields">
      {fields.map(([key, label]) => {
        const value = Math.round(bbox[key] * 100) / 100
        return (
          <label key={key} className="field field--compact">
            <span>{label}</span>
            <input
              // Новый key пересоздаёт поле, когда рамка меняется извне (например, мышью).
              key={value}
              type="number"
              step="any"
              defaultValue={value}
              onBlur={(event) => {
                const next = Number(event.target.value)
                const valid = Number.isFinite(next) && (key === 'x' || key === 'y' || next > 0)
                if (valid && next !== value) onChange({ ...bbox, [key]: next })
                else event.target.value = String(value)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') event.currentTarget.blur()
              }}
            />
          </label>
        )
      })}
    </div>
  )
}

function BrokenDrawing({ drawing, onReprocessed }: { drawing: Drawing; onReprocessed: () => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function reprocess() {
    setBusy(true)
    setError(null)
    try {
      const result = await api.reprocessDrawing(drawing.id)
      if (result.status === 'ready') onReprocessed()
      else setError(result.error)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <PageHeader title={drawing.name} />
      <Notice kind="error">
        Чертёж не удалось подготовить к просмотру: {error ?? drawing.error}
      </Notice>
      <div className="actions">
        <button type="button" className="btn btn--primary" disabled={busy} onClick={() => void reprocess()}>
          Повторить обработку
        </button>
        <a className="btn" href={drawing.original_url}>
          Скачать оригинал
        </a>
        <Link className="btn btn--ghost" to="/drawings">
          ← К списку чертежей
        </Link>
      </div>
    </div>
  )
}
