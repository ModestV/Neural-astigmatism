import {
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent,
  type Ref,
} from 'react'
import type { BBox } from '../api/types.ts'
import { isTypingTarget } from '../lib/dom.ts'
import { formatNumber, unitsLabel } from '../lib/format.ts'
import { clamp, rectFromPoints } from '../lib/geometry.ts'

export type Tool = 'pan' | 'select'

/** Рамка поверх чертежа: найденный экземпляр или образец из библиотеки. */
export interface Overlay {
  id: string
  bbox: BBox
  kind: 'match' | 'template'
  label?: string
}

export interface ViewerHandle {
  fit: () => void
  zoomBy: (factor: number) => void
  zoomTo: (bbox: BBox) => void
}

interface Props {
  ref?: Ref<ViewerHandle>
  /** Страница в SVG */
  src: string
  /** Размер страницы в единицах страницы */
  width: number
  height: number
  units: string
  tool: Tool
  selection: BBox | null
  onSelectionChange: (bbox: BBox | null) => void
  overlays?: Overlay[]
  highlightedId?: string | null
  onImageLoad?: () => void
}

/** Экранная точка = точка страницы × zoom + (x, y). */
interface View {
  zoom: number
  x: number
  y: number
}

type Drag =
  | { kind: 'pan'; startX: number; startY: number; view: View }
  | { kind: 'select'; x: number; y: number }

const MIN_SELECTION_PX = 4
/** Во сколько раз можно приблизить относительно режима «по размеру окна». */
const MAX_ZOOM_IN = 64
const FIT_PADDING = 0.95

function fitZoomOf(el: HTMLElement, width: number, height: number): number {
  if (!el.clientWidth || !el.clientHeight) return 1
  return Math.min(el.clientWidth / width, el.clientHeight / height) * FIT_PADDING
}

/** Вид, при котором лист целиком помещается в окно и стоит по центру. */
function fitView(el: HTMLElement, width: number, height: number): View {
  const zoom = fitZoomOf(el, width, height)
  return { zoom, x: (el.clientWidth - width * zoom) / 2, y: (el.clientHeight - height * zoom) / 2 }
}

/**
 * Просмотрщик страницы чертежа: масштаб колесом мыши, перемещение (инструмент «Рука»,
 * пробел или средняя кнопка), выделение рамкой, подсветка найденного.
 *
 * Страница — это <img> с SVG. Её размер меняется через width/height, а не через
 * CSS transform: так браузер перерисовывает вектор и линии остаются чёткими при любом
 * масштабе. Рамки рисуются в отдельном <svg> с viewBox в единицах страницы.
 */
export function DrawingViewer({
  ref,
  src,
  width,
  height,
  units,
  tool,
  selection,
  onSelectionChange,
  overlays = [],
  highlightedId = null,
  onImageLoad,
}: Props) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<Drag | null>(null)
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 })
  const [draft, setDraft] = useState<BBox | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [panning, setPanning] = useState(false)
  const [spaceDown, setSpaceDown] = useState(false)
  const [loaded, setLoaded] = useState<{ src: string; ok: boolean } | null>(null)
  const status = loaded?.src !== src ? 'loading' : loaded.ok ? 'ready' : 'error'

  function clampZoom(zoom: number): number {
    const el = viewportRef.current
    const fitZoom = el ? fitZoomOf(el, width, height) : 1
    return clamp(zoom, fitZoom / 4, fitZoom * MAX_ZOOM_IN)
  }

  function fit() {
    const el = viewportRef.current
    if (el) setView(fitView(el, width, height))
  }

  /** Масштаб вокруг точки (sx, sy) окна просмотра: эта точка остаётся на месте. */
  function zoomAt(sx: number, sy: number, factor: number) {
    setView((v) => {
      const zoom = clampZoom(v.zoom * factor)
      const k = zoom / v.zoom
      return { zoom, x: sx - (sx - v.x) * k, y: sy - (sy - v.y) * k }
    })
  }

  function zoomBy(factor: number) {
    const el = viewportRef.current
    if (el) zoomAt(el.clientWidth / 2, el.clientHeight / 2, factor)
  }

  function zoomTo(bbox: BBox) {
    const el = viewportRef.current
    if (!el) return
    // Объект займёт около четверти окна — видно и его, и то, что вокруг.
    const zoom = clampZoom(Math.min(el.clientWidth / (bbox.w * 4), el.clientHeight / (bbox.h * 4)))
    setView({
      zoom,
      x: el.clientWidth / 2 - (bbox.x + bbox.w / 2) * zoom,
      y: el.clientHeight / 2 - (bbox.y + bbox.h / 2) * zoom,
    })
  }

  useImperativeHandle(ref, () => ({ fit, zoomBy, zoomTo }))

  // Новая страница (src) — показываем её целиком. Размер окна известен только после
  // отрисовки, поэтому вид выставляется в layout-эффекте, до того как кадр попадёт на экран.
  useLayoutEffect(() => {
    const el = viewportRef.current
    if (el) setView(fitView(el, width, height))
  }, [src, width, height])

  // React подписывается на wheel как passive, и preventDefault там не срабатывает.
  // Вешаем обработчик сами, иначе вместе с масштабом прокручивалась бы вся страница.
  const zoomAtRef = useRef(zoomAt)
  useLayoutEffect(() => {
    zoomAtRef.current = zoomAt
  })
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = el.getBoundingClientRect()
      const lines = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1
      const delta = clamp(event.deltaY * lines, -200, 200)
      zoomAtRef.current(event.clientX - rect.left, event.clientY - rect.top, Math.exp(-delta * 0.002))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Пока зажат пробел, любой инструмент работает как «Рука».
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !isTypingTarget(event.target)) {
        event.preventDefault()
        setSpaceDown(true)
      }
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      if (!isTypingTarget(event.target)) event.preventDefault()
      setSpaceDown(false)
    }
    const onBlur = () => setSpaceDown(false)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [])

  function pagePoint(event: { clientX: number; clientY: number }) {
    const rect = viewportRef.current!.getBoundingClientRect()
    return {
      x: (event.clientX - rect.left - view.x) / view.zoom,
      y: (event.clientY - rect.top - view.y) / view.zoom,
    }
  }

  function selectionTo(event: PointerEvent, drag: { x: number; y: number }): BBox {
    const p = pagePoint(event)
    return rectFromPoints(drag.x, drag.y, clamp(p.x, 0, width), clamp(p.y, 0, height))
  }

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (status !== 'ready') return
    const pan = event.button === 1 || (event.button === 0 && (tool === 'pan' || spaceDown))
    const select = event.button === 0 && tool === 'select' && !spaceDown
    if (!pan && !select) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    if (pan) {
      dragRef.current = { kind: 'pan', startX: event.clientX, startY: event.clientY, view }
      setPanning(true)
    } else {
      const p = pagePoint(event)
      dragRef.current = { kind: 'select', x: clamp(p.x, 0, width), y: clamp(p.y, 0, height) }
    }
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const p = pagePoint(event)
    const inside = p.x >= 0 && p.y >= 0 && p.x <= width && p.y <= height
    setCursor(inside ? p : null)

    const drag = dragRef.current
    if (drag?.kind === 'pan') {
      setView({
        ...drag.view,
        x: drag.view.x + event.clientX - drag.startX,
        y: drag.view.y + event.clientY - drag.startY,
      })
    } else if (drag?.kind === 'select') {
      setDraft(selectionTo(event, drag))
    }
  }

  function onPointerUp(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current
    dragRef.current = null
    setPanning(false)
    if (drag?.kind !== 'select') return
    const rect = selectionTo(event, drag)
    setDraft(null)
    // Простой щелчок без протягивания снимает выделение.
    const bigEnough = rect.w * view.zoom >= MIN_SELECTION_PX && rect.h * view.zoom >= MIN_SELECTION_PX
    onSelectionChange(bigEnough ? rect : null)
  }

  function onPointerCancel() {
    dragRef.current = null
    setPanning(false)
    setDraft(null)
  }

  const shown = draft ?? selection
  const mode = panning ? 'is-panning' : tool === 'pan' || spaceDown ? 'is-pan' : 'is-select'
  const unit = unitsLabel(units)

  return (
    <div className="viewer">
      <div
        ref={viewportRef}
        className={`viewer__viewport ${mode}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onPointerLeave={() => setCursor(null)}
      >
        <div
          className="viewer__stage"
          style={{
            width: width * view.zoom,
            height: height * view.zoom,
            transform: `translate(${view.x}px, ${view.y}px)`,
          }}
        >
          <img
            className="viewer__image"
            src={src}
            alt=""
            draggable={false}
            onLoad={() => {
              setLoaded({ src, ok: true })
              onImageLoad?.()
            }}
            onError={() => setLoaded({ src, ok: false })}
          />
          <svg
            className="viewer__overlay"
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="none"
          >
            {overlays.map((overlay) => (
              <g
                key={overlay.id}
                className={`overlay overlay--${overlay.kind}${
                  overlay.id === highlightedId ? ' is-highlighted' : ''
                }`}
              >
                <rect
                  x={overlay.bbox.x}
                  y={overlay.bbox.y}
                  width={overlay.bbox.w}
                  height={overlay.bbox.h}
                  vectorEffect="non-scaling-stroke"
                />
                {overlay.label && (
                  <text
                    x={overlay.bbox.x}
                    y={overlay.bbox.y - 4 / view.zoom}
                    fontSize={11 / view.zoom}
                    strokeWidth={3 / view.zoom}
                  >
                    {overlay.label}
                  </text>
                )}
              </g>
            ))}
            {shown && (
              <rect
                className={draft ? 'selection selection--draft' : 'selection'}
                x={shown.x}
                y={shown.y}
                width={shown.w}
                height={shown.h}
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
        </div>
        {status !== 'ready' && (
          <div className={`viewer__message${status === 'error' ? ' is-error' : ''}`}>
            {status === 'loading' ? 'Загрузка страницы…' : 'Не удалось загрузить страницу'}
          </div>
        )}
      </div>
      <div className="viewer__status">
        <span>
          {cursor ? `x ${formatNumber(cursor.x)}, y ${formatNumber(cursor.y)} ${unit}` : 'Курсор вне листа'}
        </span>
        <span>Масштаб {Math.round(view.zoom * 100)}%</span>
        <span>
          Лист {formatNumber(width)} × {formatNumber(height)} {unit}
        </span>
        {shown && (
          <span>
            Рамка {formatNumber(shown.w)} × {formatNumber(shown.h)} {unit}
          </span>
        )}
      </div>
    </div>
  )
}
