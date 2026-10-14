import { Link } from 'react-router'
import { PageHeader } from '../components/ui.tsx'

export function NotFoundPage() {
  return (
    <div className="page">
      <PageHeader title="Страница не найдена" />
      <Link to="/">На главную</Link>
    </div>
  )
}
