import { useState, type DragEvent } from 'react'
import { api, errorText } from '../api/client.ts'
import type { Drawing } from '../api/types.ts'
import { plural } from '../lib/format.ts'

const ACCEPT = '.pdf,.svg,.dxf,.dwg'

interface UploadItem {
  key: number
  name: string
  status: 'uploading' | 'ok' | 'warning' | 'error'
  message?: string
}

let nextKey = 0

export function UploadDropzone({ onUploaded }: { onUploaded: (drawing: Drawing) => void }) {
  const [dragOver, setDragOver] = useState(false)
  const [items, setItems] = useState<UploadItem[]>([])

  const update = (key: number, patch: Partial<UploadItem>) =>
    setItems((list) => list.map((item) => (item.key === key ? { ...item, ...patch } : item)))

  async function upload(files: File[]) {
    // Грузим по одному: файлы могут быть большими, а сервер обрабатывает их сразу.
    for (const file of files) {
      const key = nextKey++
      setItems((list) => [{ key, name: file.name, status: 'uploading' }, ...list])
      try {
        const drawing = await api.uploadDrawing(file)
        if (drawing.status === 'ready') {
          update(key, {
            status: 'ok',
            message: plural(drawing.pages.length, ['лист', 'листа', 'листов']),
          })
        } else {
          update(key, {
            status: 'warning',
            message: `загружен, но не обработан: ${drawing.error}`,
          })
        }
        onUploaded(drawing)
      } catch (error) {
        update(key, { status: 'error', message: errorText(error) })
      }
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragOver(false)
    void upload(Array.from(event.dataTransfer.files))
  }

  return (
    <div className="upload">
      <label
        className={`dropzone${dragOver ? ' is-over' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(event) => {
            void upload(Array.from(event.target.files ?? []))
            event.target.value = ''
          }}
        />
        <strong>Перетащите чертежи сюда или нажмите, чтобы выбрать файлы</strong>
        <span className="muted">PDF, SVG, DXF или DWG (для DWG нужен конвертер на сервере)</span>
      </label>
      {items.length > 0 && (
        <ul className="upload__list">
          {items.map((item) => (
            <li key={item.key} className={`upload__item upload__item--${item.status}`}>
              <span>{item.name}</span>
              <span className="muted">
                {item.status === 'uploading' ? 'загрузка и обработка…' : item.message}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
