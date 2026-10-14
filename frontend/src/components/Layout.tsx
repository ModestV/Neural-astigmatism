import { NavLink, Outlet } from 'react-router'

const NAV = [
  { to: '/', label: 'Главная', end: true },
  { to: '/drawings', label: 'Чертежи' },
  { to: '/symbols', label: 'Библиотека' },
  { to: '/search', label: 'Поиск' },
  { to: '/evaluation', label: 'Оценка качества' },
  { to: '/about', label: 'О проекте' },
]

export function Layout() {
  return (
    <div className="app">
      <header className="app-header">
        <NavLink to="/" className="app-logo">
          Symbol Search
        </NavLink>
        <nav className="app-nav">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>
              {item.label}
            </NavLink>
          ))}
          <a href="/docs" target="_blank" rel="noreferrer">
            API ↗
          </a>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}
