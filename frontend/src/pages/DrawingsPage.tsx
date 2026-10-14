import { useState } from 'react'
import { Link } from 'react-router'
import { api, errorText } from '../api/client.ts'
import type { Drawing } from '../api/types.ts'
import { UploadDropzone } from '../components/UploadDropzone.tsx'
import { Empty, ErrorMessage, Loading, Notice, PageHeader } from '../components/ui.tsx'
import { useAsync } from '../hooks/useAsync.ts'
import { FORMAT_LABELS, formatBytes, formatDate, plural } from '../lib/format.ts'

export function DrawingsPage() {
  const drawings = useAsync(() => api.listDrawings())
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function remove(drawing: Drawing) {
    const question = `Удалить чертёж «${drawing.name}»? Обозначения, выделенные на нём, останутся в библиотеке.`
    if (!window.confirm(question)) return
    await act(drawing, () => api.deleteDrawing(drawing.id))
  }

  async function act(drawing: Drawing, action: () => Promise<unknown>) {
    setBusyId(drawing.id)
    setError(null)
    try {
      await action()
      drawings.reload()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusyId(null)
    }
  }

  const list = drawings.data ?? []

  return (
    <div className="page">
      <PageHeader
        title="Чертежи"
        description="Загруженные чертежи. Откройте чертёж, чтобы рассмотреть его и выделить условные обозначения."
      />
      <UploadDropzone onUploaded={drawings.reload} />
      {error && <Notice kind="error">{error}</Notice>}
      {drawings.error && <ErrorMessage error={drawings.error} onRetry={drawings.reload} />}
      {drawings.loading && !drawings.data && <Loading />}
      {drawings.data && list.length === 0 && (
        <Empty>
          Чертежей пока нет. Для пробы загрузите файлы из папки <code>samples/</code> в корне
          проекта.
        </Empty>
      )}
      {list.length > 0 && (
        <ul className="card-grid">
          {list.map((drawing) => (
            <DrawingCard
              key={drawing.id}
              drawing={drawing}
              busy={busyId === drawing.id}
              onDelete={() => void remove(drawing)}
              onReprocess={() => void act(drawing, () => api.reprocessDrawing(drawing.id))}
            />
          ))}
        </ul>
      )}
    </div>
  )
}

function DrawingCard(props: {
  drawing: Drawing
  busy: boolean
  onDelete: () => void
  onReprocess: () => void
}) {
  const { drawing, busy, onDelete, onReprocess } = props
  const ready = drawing.status === 'ready'
  const href = `/drawings/${drawing.id}`
  const thumb = drawing.pages[0]?.thumb_url

  return (
    <li className="card">
      {ready ? (
        <Link to={href} className="card__thumb">
          {thumb && <img src={thumb} alt="" loading="lazy" />}
        </Link>
      ) : (
        <div className="card__thumb card__thumb--empty">Нет превью</div>
      )}
      <div className="card__body">
        {ready ? (
          <Link to={href} className="card__title" title={drawing.name}>
            {drawing.name}
          </Link>
        ) : (
          <span className="card__title" title={drawing.name}>
            {drawing.name}
          </span>
        )}
        <div className="card__meta">
          <span className="badge">{FORMAT_LABELS[drawing.format]}</span>
          {ready && <span>{plural(drawing.pages.length, ['лист', 'листа', 'листов'])}</span>}
          <span>{formatBytes(drawing.size_bytes)}</span>
          <span>{formatDate(drawing.created_at)}</span>
        </div>
        {!ready && <p className="card__error">{drawing.error}</p>}
        <div className="actions">
          {ready ? (
            <Link className="btn btn--small btn--primary" to={href}>
              Открыть
            </Link>
          ) : (
            <button type="button" className="btn btn--small" disabled={busy} onClick={onReprocess}>
              Повторить обработку
            </button>
          )}
          <a className="btn btn--small" href={drawing.original_url}>
            Скачать
          </a>
          <button
            type="button"
            className="btn btn--small btn--danger"
            disabled={busy}
            onClick={onDelete}
          >
            Удалить
          </button>
        </div>
      </div>
    </li>
  )
}
