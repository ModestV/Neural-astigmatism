import type {
  Drawing,
  Health,
  LibrarySymbol,
  RunSummary,
  SearchRequest,
  SearchRun,
  Stats,
  SymbolCreate,
  SymbolUpdate,
} from './types.ts'

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, init)
  } catch {
    throw new ApiError('Сервер недоступен — запущен ли бэкенд?', 0)
  }
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const detail = ((await response.json()) as { detail?: unknown }).detail
    if (typeof detail === 'string') {
      return detail
    }
    if (Array.isArray(detail)) {
      // ошибки валидации FastAPI: [{ loc, msg, type }, …]
      return detail.map((item: { msg?: string }) => item.msg ?? '').join('; ')
    }
  } catch {
    // Тело не JSON. Так бывает, когда прокси Vite не может достучаться до бэкенда.
    if (response.status >= 500) {
      return 'Сервер не отвечает — запущен ли бэкенд?'
    }
  }
  return `Ошибка ${response.status} ${response.statusText}`
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      search.set(key, String(value))
    }
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

const path = encodeURIComponent

export const api = {
  health: () => request<Health>('/api/health'),
  stats: () => request<Stats>('/api/stats'),

  listDrawings: () => request<Drawing[]>('/api/drawings'),
  getDrawing: (id: string) => request<Drawing>(`/api/drawings/${path(id)}`),
  uploadDrawing: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Drawing>('/api/drawings', { method: 'POST', body: form })
  },
  reprocessDrawing: (id: string) =>
    request<Drawing>(`/api/drawings/${path(id)}/reprocess`, { method: 'POST' }),
  deleteDrawing: (id: string) =>
    request<void>(`/api/drawings/${path(id)}`, { method: 'DELETE' }),

  listSymbols: (params: { drawing_id?: string; category?: string } = {}) =>
    request<LibrarySymbol[]>(`/api/symbols${query(params)}`),
  createSymbol: (body: SymbolCreate) =>
    request<LibrarySymbol>('/api/symbols', json('POST', body)),
  updateSymbol: (id: string, patch: SymbolUpdate) =>
    request<LibrarySymbol>(`/api/symbols/${path(id)}`, json('PATCH', patch)),
  deleteSymbol: (id: string) =>
    request<void>(`/api/symbols/${path(id)}`, { method: 'DELETE' }),

  search: (body: SearchRequest) => request<SearchRun>('/api/search', json('POST', body)),
  listRuns: (params: { drawing_id?: string; limit?: number } = {}) =>
    request<RunSummary[]>(`/api/runs${query(params)}`),
  getRun: (id: string) => request<SearchRun>(`/api/runs/${path(id)}`),
}

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
