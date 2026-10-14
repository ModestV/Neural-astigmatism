import { Link } from 'react-router'
import { api } from '../api/client.ts'
import { ErrorMessage, Loading, PageHeader } from '../components/ui.tsx'
import { useAsync } from '../hooks/useAsync.ts'

export function HomePage() {
  const stats = useAsync(() => api.stats())
  const health = useAsync(() => api.health())

  return (
    <div className="page">
      <PageHeader
        title="Поиск условных обозначений на чертежах"
        description="Стартовая версия сервиса: просмотр чертежей, выделение обозначений, библиотека образцов. Алгоритм поиска пока заменён заглушкой."
      />

      <section className="stats">
        <Stat to="/drawings" label="Чертежи" value={stats.data?.drawings} />
        <Stat to="/symbols" label="Обозначения в библиотеке" value={stats.data?.symbols} />
        <Stat to="/search" label="Запуски поиска" value={stats.data?.runs} />
      </section>

      <section>
        <h2>Как пользоваться</h2>
        <ol className="steps">
          <li>
            <Link to="/drawings">Загрузите чертёж</Link> в формате PDF, SVG, DXF или DWG.
          </li>
          <li>Откройте его и обведите условное обозначение инструментом «Рамка».</li>
          <li>
            Сохраните обозначение в <Link to="/symbols">библиотеку</Link> или сразу нажмите
            «Найти похожие».
          </li>
          <li>
            Чтобы искать обозначения из библиотеки сразу на нескольких чертежах, откройте раздел{' '}
            <Link to="/search">«Поиск»</Link>.
          </li>
        </ol>
      </section>

      <section>
        <h2>Состояние сервиса</h2>
        {health.error && <ErrorMessage error={health.error} onRetry={health.reload} />}
        {health.loading && !health.data && <Loading />}
        {health.data && (
          <dl className="dl">
            <dt>Версия API</dt>
            <dd>{health.data.version}</dd>
            <dt>Алгоритм поиска</dt>
            <dd>
              «{health.data.detector.name}» — {health.data.detector.description}
            </dd>
            <dt>Форматы</dt>
            <dd>
              PDF, SVG, DXF
              {health.data.dwg_converter
                ? `, DWG (через ${health.data.dwg_converter})`
                : '; для DWG нужен конвертер (ODA File Converter или LibreDWG)'}
            </dd>
          </dl>
        )}
      </section>
    </div>
  )
}

function Stat({ to, label, value }: { to: string; label: string; value: number | undefined }) {
  return (
    <Link to={to} className="stat">
      <strong>{value ?? '—'}</strong>
      <span>{label}</span>
    </Link>
  )
}
