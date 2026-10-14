import type { SearchParams } from '../api/types.ts'

/** Параметры поиска. Пока подключена заглушка, они ни на что не влияют. */
export function SearchParamsForm(props: {
  value: SearchParams
  onChange: (value: SearchParams) => void
}) {
  const { value, onChange } = props
  const toggle = (key: 'rotations' | 'mirror' | 'scale') => (
    <input
      type="checkbox"
      checked={value[key]}
      onChange={(event) => onChange({ ...value, [key]: event.target.checked })}
    />
  )

  return (
    <div className="params">
      <label className="field">
        <span>Порог сходства: {Math.round(value.threshold * 100)}%</span>
        <input
          type="range"
          min={0.5}
          max={1}
          step={0.01}
          value={value.threshold}
          onChange={(event) => onChange({ ...value, threshold: Number(event.target.value) })}
        />
      </label>
      <label className="check">
        {toggle('rotations')} Искать повёрнутые
      </label>
      <label className="check">
        {toggle('mirror')} Искать зеркальные
      </label>
      <label className="check">
        {toggle('scale')} Искать в другом масштабе
      </label>
    </div>
  )
}
