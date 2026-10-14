import { useState } from 'react'
import { Link } from 'react-router'
import { api, errorText } from '../api/client.ts'
import type { LibrarySymbol } from '../api/types.ts'
import { Empty, ErrorMessage, Loading, Notice, PageHeader } from '../components/ui.tsx'
import { useAsync } from '../hooks/useAsync.ts'
import { formatDate, formatNumber } from '../lib/format.ts'

export function SymbolsPage() {
  const symbols = useAsync(() => api.listSymbols())
  const [text, setText] = useState('')
  const [category, setCategory] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const all = symbols.data ?? []
  const categories = [...new Set(all.flatMap((s) => (s.category ? [s.category] : [])))].sort(
    (a, b) => a.localeCompare(b, 'ru'),
  )
  const needle = text.trim().toLowerCase()
  const visible = all.filter(
    (s) => (!category || s.category === category) && s.name.toLowerCase().includes(needle),
  )

  async function run(action: () => Promise<unknown>) {
    setError(null)
    try {
      await action()
      setEditing(null)
      symbols.reload()
    } catch (e) {
      setError(errorText(e))
    }
  }

  return (
    <div className="page">
      <PageHeader
        title="Библиотека условных обозначений"
        description="Заданный набор образцов, которые ищет алгоритм. Чтобы добавить образец, откройте чертёж, обведите обозначение рамкой и нажмите «В библиотеку»."
        actions={
          <>
            <button type="button" className="btn" disabled title="В разработке">
              Импорт набора
            </button>
            <button type="button" className="btn" disabled title="В разработке">
              Экспорт датасета
            </button>
          </>
        }
      />

      <div className="filters">
        <input
          type="search"
          placeholder="Поиск по названию"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <select value={category} onChange={(event) => setCategory(event.target.value)}>
          <option value="">Все категории</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <span className="muted">
          {visible.length} из {all.length}
        </span>
      </div>

      {error && <Notice kind="error">{error}</Notice>}
      {symbols.error && <ErrorMessage error={symbols.error} onRetry={symbols.reload} />}
      {symbols.loading && !symbols.data && <Loading />}
      {symbols.data && all.length === 0 && (
        <Empty>
          Библиотека пуста. Откройте <Link to="/drawings">чертёж</Link>, выделите обозначение и
          сохраните его.
        </Empty>
      )}

      {visible.length > 0 && (
        <ul className="card-grid card-grid--small">
          {visible.map((symbol) => (
            <li key={symbol.id} className="card">
              <div className="card__thumb card__thumb--symbol">
                {symbol.preview_url ? (
                  <img src={symbol.preview_url} alt="" />
                ) : (
                  <span className="muted">Нет превью</span>
                )}
              </div>
              <div className="card__body">
                {editing === symbol.id ? (
                  <EditSymbol
                    symbol={symbol}
                    categories={categories}
                    onCancel={() => setEditing(null)}
                    onSave={(patch) => void run(() => api.updateSymbol(symbol.id, patch))}
                  />
                ) : (
                  <SymbolInfo
                    symbol={symbol}
                    onEdit={() => setEditing(symbol.id)}
                    onDelete={() => {
                      if (window.confirm(`Удалить «${symbol.name}» из библиотеки?`)) {
                        void run(() => api.deleteSymbol(symbol.id))
                      }
                    }}
                  />
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function SymbolInfo(props: { symbol: LibrarySymbol; onEdit: () => void; onDelete: () => void }) {
  const { symbol, onEdit, onDelete } = props
  const href = `/drawings/${symbol.drawing_id}?page=${symbol.page}&symbol=${symbol.id}`
  return (
    <>
      <strong className="card__title" title={symbol.name}>
        {symbol.name}
      </strong>
      <div className="card__meta">
        {symbol.category && <span className="badge">{symbol.category}</span>}
        <span>
          {formatNumber(symbol.bbox.w)} × {formatNumber(symbol.bbox.h)}
        </span>
        <span>{formatDate(symbol.created_at)}</span>
      </div>
      <p className="small muted">
        {symbol.drawing_name ? (
          <>
            {symbol.drawing_name}, лист {symbol.page}
          </>
        ) : (
          'Исходный чертёж удалён'
        )}
      </p>
      <div className="actions">
        {symbol.drawing_name && (
          <Link className="btn btn--small" to={href}>
            На чертеже
          </Link>
        )}
        <button type="button" className="btn btn--small" onClick={onEdit}>
          Изменить
        </button>
        <button type="button" className="btn btn--small btn--danger" onClick={onDelete}>
          Удалить
        </button>
      </div>
    </>
  )
}

function EditSymbol(props: {
  symbol: LibrarySymbol
  categories: string[]
  onSave: (patch: { name: string; category: string | null }) => void
  onCancel: () => void
}) {
  const [name, setName] = useState(props.symbol.name)
  const [category, setCategory] = useState(props.symbol.category ?? '')
  const listId = `categories-${props.symbol.id}`

  return (
    <form
      className="save-form"
      onSubmit={(event) => {
        event.preventDefault()
        props.onSave({ name: name.trim(), category: category.trim() || null })
      }}
    >
      <label className="field">
        <span>Название</span>
        <input autoFocus required value={name} onChange={(event) => setName(event.target.value)} />
      </label>
      <label className="field">
        <span>Категория</span>
        <input list={listId} value={category} onChange={(event) => setCategory(event.target.value)} />
        <datalist id={listId}>
          {props.categories.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
      </label>
      <div className="actions">
        <button type="submit" className="btn btn--small btn--primary" disabled={!name.trim()}>
          Сохранить
        </button>
        <button type="button" className="btn btn--small btn--ghost" onClick={props.onCancel}>
          Отмена
        </button>
      </div>
    </form>
  )
}
