import { FreakSeekLogo, FreakSeekWordmark } from '@deepseek-ai/dsh-client-ui-primitives'
import type { SidebarBrandMarkOwnerProps } from '@deepseek-ai/dsh-client-ui-sidebar/client'

/**
 * Render the official mark with the presentation requested by its host surface.
 * @param props - Host-supplied mark presentation.
 * @returns the FreakSeek mark.
 */
export function OfficialBrandMark({ size }: SidebarBrandMarkOwnerProps) {
  return <FreakSeekLogo size={size} />
}

/**
 * Render the official name artwork without its independently slotted mark.
 * @returns the FreakSeek product name.
 */
export function OfficialBrandName() {
  return <FreakSeekWordmark />
}
