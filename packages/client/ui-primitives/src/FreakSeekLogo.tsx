import type { IconProps } from './icons/props.ts'

/**
 * The FreakSeek mark in one path: a rounded square with the "F" and spark cut out (even-odd fill),
 * on the 24-unit grid of `branding/logo.svg`.
 */
export const FREAKSEEK_LOGO_PATH = 'M6 1H18A5 5 0 0 1 23 6V18A5 5 0 0 1 18 23H6A5 5 0 0 1 1 18V6A5 5 0 0 1 6 1Z'
  + 'M6.5 5H17.5V8H9.5V10.5H14.5V13.5H9.5V19H6.5Z'
  + 'M16.6 12L12.8 17.2H15.2L13.8 21L18.4 15.4H16L17.4 12Z'

/**
 * Render the FreakSeek mark in the current text colour.
 * @param props.size - width and height in px (default 24).
 * @param props.className - extra class for layout placement.
 * @returns the mark svg (aria-hidden; pair with the product name for accessibility).
 */
export function FreakSeekLogo({ size = 24, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d={FREAKSEEK_LOGO_PATH} fill="currentColor" fillRule="evenodd" />
    </svg>
  )
}

/** The product name, a brand identifier that locales do not translate. */
export const FREAKSEEK_PRODUCT_NAME = 'FreakSeek'

/**
 * Render the FreakSeek product name as text in the inherited font.
 * @param props.className - extra class for layout placement.
 * @returns the product name span.
 */
export function FreakSeekWordmark({ className }: { readonly className?: string | undefined }) {
  return <span className={className}>{FREAKSEEK_PRODUCT_NAME}</span>
}
