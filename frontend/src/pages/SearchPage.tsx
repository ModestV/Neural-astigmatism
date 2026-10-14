import { useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import { api, errorText } from '../api/client.ts'
import type { Drawing, RunSummary, SearchParams, SearchRun } from '../api/types.ts'
import { SearchParamsForm } from '../components/SearchParamsForm.tsx'
import { Empty, ErrorMessage, Loading, Notice, PageHeader } from '../components/ui.tsx'
import { useAsync } from '../hooks/useAsync.ts'
import { formatDate, formatNumber, plural } from '../lib/format.ts'
import { DEFAULT_SEARCH_PARAMS } from '../lib/search.ts'

interface BatchResult {
  drawing: Drawing
  run?: SearchRun
  error?: string
}

export function SearchPage() {
  const drawings = useAsync(() => api.listDrawings())
  const symbols = useAsync(() => api.listSymbols())
  const runs = useAsync(() => api.listRuns({ limit: 30 }))
  const health = useAsync(() => api.health())

  const [symbolIds, setSymbolIds] = useState<Set<string>>(() => new Set())
  const [drawingIds, setDrawingIds] = useState<Set<string>>(() => new Set())
  const [params, setParams] = useState<SearchParams>(DEFAULT_SEARCH_PARAMS)
  const [results, setResults] = useState<BatchResult[]>([])
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)

  const readyDrawings = (drawings.data ?? []).filter((d) => d.status === 'ready')
  const running = progress !== null && progress.done < progress.total

  async function start() {
    const targets = readyDrawings.filter((d) => drawingIds.has(d.id))
    setResults([])
    setProgress({ done: 0, total: targets.length })
    for (const [index, drawing] of targets.entries()) {
      try {
        const run = await api.search({
          drawing_id: drawing.id,
          pages: null,
          query: null,
          symbol_ids: [...symbolIds],
          params,
        })
        setResults((prev) => [...prev, { drawing, run }])
      } catch (error) {
        setResults((prev) => [...prev, { drawing, error: errorText(error) }])
      }
      setProgress({ done: index + 1, total: targets.length })
    }
    runs.reload()
  }

  return (
    <div className="page">
      <PageHeader
        title="Пакетный поиск"
        description="Поиск обозначений из библиотеки сразу на нескольких чертежах. Найденное можно открыть на чертеже."
      />
      {health.data && (
        <Notice kind="warning">
          Алгоритм поиска: «{health.data.detector.name}» — {health.data.detector.description}.
        </Notice>
      )}

      <div className="columns">
        <section className="panel">
          <h2>1. Что искать</h2>
          {symbols.error && <ErrorMessage error={symbols.error} onRetry={symbols.reload} />}
          <CheckList
            items={(symbols.data ?? []).map((s) => ({
              id: s.id,
              label: s.name,
              hint: s.category ?? undefined,
            }))}
            selected={symbolIds}
            onChange={setSymbolIds}
            empty={
              <>
                Библиотека пуста — <Link to="/drawings">выделите обозначения на чертежах</Link>.
              </>
            }
          />
        </section>
        <section className="panel">
          <h2>2. Где искать</h2>
          {drawings.error && <ErrorMessage error={drawings.error} onRetry={drawings.reload} />}
          <CheckList
            items={readyDrawings.map((d) => ({
              id: d.id,
              label: d.name,
              hint: plural(d.pages.length, ['лист', 'листа', 'листов']),
            }))}
            selected={drawingIds}
            onChange={setDrawingIds}
            empty={
              <>
                Чертежей нет — <Link to="/drawings">загрузите их</Link>.
              </>
            }
          />
        </section>
        <section className="panel">
          <h2>3. Параметры</h2>
          <SearchParamsForm value={params} onChange={setParams} />
          <button
            type="button"
            className="btn btn--primary"
            disabled={running || symbolIds.size === 0 || drawingIds.size === 0}
            onClick={() => void start()}
          >
            {running && progress ? `Поиск… ${progress.done} из ${progress.total}` : 'Запустить поиск'}
          </button>
        </section>
      </div>

      {results.length > 0 && (
        <section>
          <h2>Результаты</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Чертёж</th>
                <th>Найдено</th>
                <th>Время</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {results.map(({ drawing, run, error }) => (
                <tr key={drawing.id}>
                  <td>{drawing.name}</td>
                  {run ? (
                    <>
                      <td>{run.detections.length}</td>
                      <td>{formatNumber(run.duration_ms, 0)} мс</td>
                      <td>
                        <Link to={`/drawings/${drawing.id}?run=${run.id}`}>Открыть</Link>
                      </td>
                    </>
                  ) : (
                    <td colSpan={3} className="error-text">
                      {error}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section>
        <h2>История запусков</h2>
        {runs.error && <ErrorMessage error={runs.error} onRetry={runs.reload} />}
        {runs.loading && !runs.data && <Loading />}
        {runs.data?.length === 0 && <Empty>Поиск ещё не запускали.</Empty>}
        {runs.data && runs.data.length > 0 && (
          <table className="table">
            <thead>
              <tr>
                <th>Когда</th>
                <th>Чертёж</th>
                <th>Образец</th>
                <th>Алгоритм</th>
                <th>Найдено</th>
                <th>Время</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.data.map((run) => (
                <tr key={run.id}>
                  <td>{formatDate(run.created_at)}</td>
                  <td>{run.drawing_name ?? <span className="muted">удалён</span>}</td>
                  <td>{describeQuery(run)}</td>
                  <td>
                    {run.detector}
                    {run.status === 'failed' && <span className="error-text"> · ошибка</span>}
                  </td>
                  <td>{run.detections_count}</td>
                  <td>{formatNumber(run.duration_ms, 0)} мс</td>
                  <td>
                    {run.drawing_name && (
                      <Link to={`/drawings/${run.drawing_id}?run=${run.id}`}>Открыть</Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}

function describeQuery(run: RunSummary): string {
  const parts = []
  if (run.has_region_query) parts.push('фрагмент')
  if (run.symbols_count > 0) {
    parts.push(plural(run.symbols_count, ['обозначение', 'обозначения', 'обозначений']))
  }
  return parts.join(' + ')
}

function CheckList(props: {
  items: { id: string; label: string; hint?: string }[]
  selected: Set<string>
  onChange: (selected: Set<string>) => void
  empty: ReactNode
}) {
  const { items, selected, onChange } = props
  if (items.length === 0) return <p className="muted">{props.empty}</p>

  const allSelected = items.every((item) => selected.has(item.id))
  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    onChange(next)
  }

  return (
    <>
      <label className="check">
        <input
          type="checkbox"
          checked={allSelected}
          onChange={() => onChange(allSelected ? new Set() : new Set(items.map((item) => item.id)))}
        />
        <strong>Выбрать все</strong>
      </label>
      <ul className="checklist">
        {items.map((item) => (
          <li key={item.id}>
            <label className="check">
              <input
                type="checkbox"
                checked={selected.has(item.id)}
                onChange={() => toggle(item.id)}
              />
              <span>{item.label}</span>
              {item.hint && <span className="muted small">{item.hint}</span>}
            </label>
          </li>
        ))}
      </ul>
    </>
  )
}
