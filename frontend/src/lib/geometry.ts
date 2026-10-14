import type { BBox } from '../api/types.ts'

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

/** Прямоугольник по двум противоположным углам. */
export function rectFromPoints(x0: number, y0: number, x1: number, y1: number): BBox {
  return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) }
}

export function roundBBox(bbox: BBox, digits = 2): BBox {
  const k = 10 ** digits
  const r = (v: number) => Math.round(v * k) / k
  return { x: r(bbox.x), y: r(bbox.y), w: r(bbox.w), h: r(bbox.h) }
}
