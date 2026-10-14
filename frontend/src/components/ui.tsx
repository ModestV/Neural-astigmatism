// Простые общие элементы интерфейса. Оформление временное — заменится вместе с дизайном.
import type { ReactNode } from 'react'

export function PageHeader(props: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        <h1>{props.title}</h1>
        {props.description && <p className="muted">{props.description}</p>}
      </div>
      {props.actions && <div className="page-header__actions">{props.actions}</div>}
    </header>
  )
}

export function Loading({ text = 'Загрузка…' }: { text?: string }) {
  return <p className="muted">{text}</p>
}

export function ErrorMessage({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="notice notice--error" role="alert">
      {error.message}
      {onRetry && (
        <button type="button" className="btn btn--small" onClick={onRetry}>
          Повторить
        </button>
      )}
    </div>
  )
}

export type NoticeKind = 'info' | 'ok' | 'warning' | 'error'

export function Notice({ kind, children }: { kind: NoticeKind; children: ReactNode }) {
  return (
    <div className={`notice notice--${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
      {children}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

/** Заглушка раздела, который ещё не реализован. */
export function Stub({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="stub">
      <span className="badge badge--stub">Заглушка</span>
      <h2>{title}</h2>
      {children}
    </section>
  )
}
