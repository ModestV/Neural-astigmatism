import { Route, Routes } from 'react-router'
import { Layout } from './components/Layout.tsx'
import { AboutPage } from './pages/AboutPage.tsx'
import { DrawingsPage } from './pages/DrawingsPage.tsx'
import { EvaluationPage } from './pages/EvaluationPage.tsx'
import { HomePage } from './pages/HomePage.tsx'
import { NotFoundPage } from './pages/NotFoundPage.tsx'
import { SearchPage } from './pages/SearchPage.tsx'
import { SymbolsPage } from './pages/SymbolsPage.tsx'
import { ViewerPage } from './pages/ViewerPage.tsx'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="drawings" element={<DrawingsPage />} />
        <Route path="drawings/:id" element={<ViewerPage />} />
        <Route path="symbols" element={<SymbolsPage />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
