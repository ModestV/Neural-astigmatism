import type { DrawingFormat } from '../api/types.ts'

const dateFormat = new Intl.DateTimeFormat('ru-RU', { dateStyle: 'short', timeStyle: 'short' })
const pluralRules = new Intl.PluralRules('ru-RU')

export function formatDate(iso: string): string {
  return dateFormat.format(new Date(iso))
}

export function formatNumber(value: number, maxDigits = 1): string {
  return value.toLocaleString('ru-RU', { maximumFractionDigits: maxDigits })
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`
  if (bytes < 1024 * 1024) return `${formatNumber(bytes / 1024, 0)} КБ`
  return `${formatNumber(bytes / 1024 / 1024)} МБ`
}

/** plural(5, ['лист', 'листа', 'листов']) → «5 листов» */
export function plural(count: number, forms: [one: string, few: string, many: string]): string {
  const rule = pluralRules.select(count)
  const word = rule === 'one' ? forms[0] : rule === 'few' ? forms[1] : forms[2]
  return `${count} ${word}`
}

export function unitsLabel(units: string): string {
  return units === 'pt' ? 'pt' : 'ед.'
}

export const FORMAT_LABELS: Record<DrawingFormat, string> = {
  pdf: 'PDF',
  svg: 'SVG',
  dxf: 'DXF',
  dwg: 'DWG',
}
