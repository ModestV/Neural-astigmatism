import { PageHeader, Stub } from '../components/ui.tsx'

const METRICS = ['Precision', 'Recall', 'F1', 'AP@0.5', 'Время на лист']

export function EvaluationPage() {
  return (
    <div className="page">
      <PageHeader
        title="Оценка качества"
        description="Сравнение результатов алгоритма с эталонной разметкой — основа отчёта о качестве по проекту."
      />
      <Stub title="Раздел в разработке">
        <p>Что здесь будет:</p>
        <ul>
          <li>
            Загрузка эталонной разметки (ground truth) для чертежей. Пример формата —{' '}
            <code>samples/ground_truth.json</code>.
          </li>
          <li>Сопоставление найденных и эталонных рамок по IoU.</li>
          <li>Метрики по каждому обозначению и по набору в целом, время работы.</li>
          <li>Разбор ошибок: пропуски и ложные срабатывания с подсветкой на чертеже.</li>
          <li>Сравнение разных версий алгоритма между собой.</li>
        </ul>
      </Stub>

      <table className="table table--placeholder">
        <thead>
          <tr>
            <th>Обозначение</th>
            {METRICS.map((metric) => (
              <th key={metric}>{metric}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>—</td>
            {METRICS.map((metric) => (
              <td key={metric}>—</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  )
}
