import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | undefined
  error: Error | undefined
  loading: boolean
  /** Загрузить заново (например, после изменения данных). */
  reload: () => void
}

/**
 * Загружает данные при первом показе компонента и заново — когда меняется key
 * (например, id из адреса) или вызван reload(). Пока идёт повторная загрузка,
 * остаются видны старые данные.
 *
 *   const drawing = useAsync(() => api.getDrawing(id), id)
 */
export function useAsync<T>(load: () => Promise<T>, key = ''): AsyncState<T> {
  const [version, setVersion] = useState(0)
  const request = `${key}#${version}`
  const [result, setResult] = useState<{ request: string; data?: T; error?: Error } | null>(null)

  // Свежая функция загрузки без перезапуска эффекта на каждом рендере.
  const loadRef = useRef(load)
  useLayoutEffect(() => {
    loadRef.current = load
  })

  useEffect(() => {
    let cancelled = false
    loadRef.current().then(
      (data) => {
        if (!cancelled) setResult({ request, data })
      },
      (error: unknown) => {
        if (!cancelled) {
          setResult({ request, error: error instanceof Error ? error : new Error(String(error)) })
        }
      },
    )
    return () => {
      cancelled = true
    }
  }, [request])

  const reload = useCallback(() => setVersion((v) => v + 1), [])
  const loading = result?.request !== request
  return {
    data: result?.data,
    error: loading ? undefined : result?.error,
    loading,
    reload,
  }
}
