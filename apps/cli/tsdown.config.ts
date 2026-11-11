import { defineConfig } from 'tsdown'

/**
 * The dsh CLI ships its command, the FreakSeek command, and the profile lifecycle shared with Desktop.
 * Declarations come from `tsc -b` (dts: false), matching every package.
 */
export default defineConfig({
  entry: { bin: 'lib/types/installed-bin.js', freakseek: 'lib/types/freakseek-bin.js', 'profile-boot': 'lib/types/profile-boot.js' },
  outDir: 'lib',
  format: ['esm'],
  platform: 'node',
  target: 'es2024',
  fixedExtension: false,
  dts: false,
  clean: ['lib/*.js'],
})
