import type { BBox } from '../api/types.ts'

const images = new Map<string, Promise<HTMLImageElement>>()

function loadImage(src: string): Promise<HTMLImageElement> {
  let image = images.get(src)
  if (!image) {
    if (images.size > 8) images.clear()
    image = new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('Не удалось загрузить страницу'))
      img.src = src
    })
    images.set(src, image)
    image.catch(() => images.delete(src))
  }
  return image
}

/**
 * Вырезает фрагмент страницы в PNG (data URL) — превью образца.
 * SVG страницы рисуется на canvas сразу в нужном масштабе, поэтому превью чёткое
 * даже для мелких обозначений.
 */
export async function cropPage(
  src: string,
  pageWidth: number,
  pageHeight: number,
  bbox: BBox,
  maxSide = 256,
): Promise<string> {
  const img = await loadImage(src)
  // Собственный размер SVG в CSS-пикселях может не совпадать с единицами страницы.
  const kx = (img.naturalWidth || pageWidth) / pageWidth
  const ky = (img.naturalHeight || pageHeight) / pageHeight
  const scale = Math.min(maxSide / bbox.w, maxSide / bbox.h, 64)

  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(bbox.w * scale))
  canvas.height = Math.max(1, Math.round(bbox.h * scale))
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas недоступен')
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(
    img,
    bbox.x * kx,
    bbox.y * ky,
    bbox.w * kx,
    bbox.h * ky,
    0,
    0,
    canvas.width,
    canvas.height,
  )
  return canvas.toDataURL('image/png')
}
