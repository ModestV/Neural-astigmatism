// Типы данных API. Повторяют схемы бэкенда: backend/app/models.py и backend/app/schemas.py.
// Полное описание — в Swagger: http://localhost:8000/docs

/** Прямоугольник в единицах страницы: (x, y) — левый верхний угол, ось Y вниз. */
export interface BBox {
  x: number
  y: number
  w: number
  h: number
}

export type DrawingFormat = 'pdf' | 'svg' | 'dxf' | 'dwg'

export interface PageInfo {
  index: number
  width: number
  height: number
  image_url: string
  thumb_url: string
}

export interface Drawing {
  id: string
  name: string
  format: DrawingFormat
  size_bytes: number
  created_at: string
  status: 'ready' | 'error'
  error: string | null
  /** pt — пункты PDF, svg — единицы viewBox */
  units: string
  pages: PageInfo[]
  original_url: string
}

export interface LibrarySymbol {
  id: string
  name: string
  category: string | null
  drawing_id: string
  page: number
  bbox: BBox
  created_at: string
  has_preview: boolean
  preview_url: string | null
  /** null — исходный чертёж удалён */
  drawing_name: string | null
}

export interface SymbolCreate {
  name: string
  category: string | null
  drawing_id: string
  page: number
  bbox: BBox
  /** data:image/png;base64,… */
  preview_png: string | null
}

export interface SymbolUpdate {
  name?: string
  category?: string | null
}

export interface SearchParams {
  threshold: number
  rotations: boolean
  mirror: boolean
  scale: boolean
}

export interface RegionQuery {
  drawing_id: string
  page: number
  bbox: BBox
}

export interface SearchRequest {
  drawing_id: string
  /** null — все страницы */
  pages: number[] | null
  query: RegionQuery | null
  symbol_ids: string[]
  params: SearchParams
}

export interface Detection {
  page: number
  bbox: BBox
  score: number
  /** null — найден выделенный фрагмент */
  symbol_id: string | null
  angle: number
  mirrored: boolean
}

export interface SearchRun {
  id: string
  created_at: string
  request: SearchRequest
  detector: string
  status: 'done' | 'failed'
  message: string | null
  duration_ms: number
  detections: Detection[]
}

export interface RunSummary {
  id: string
  created_at: string
  drawing_id: string
  drawing_name: string | null
  detector: string
  status: 'done' | 'failed'
  message: string | null
  duration_ms: number
  detections_count: number
  has_region_query: boolean
  symbols_count: number
}

export interface Health {
  status: 'ok'
  version: string
  detector: { name: string; description: string }
  /** null — конвертер DWG не установлен */
  dwg_converter: string | null
}

export interface Stats {
  drawings: number
  symbols: number
  runs: number
}
