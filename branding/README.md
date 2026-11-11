# branding/

FreakSeek's mark: a white "F" with a spark on a violet-to-orange rounded square. [`logo.svg`](logo.svg) is the source; the in-app monochrome mark (`FreakSeekLogo` in `@deepseek-ai/dsh-client-ui-primitives`) uses the same 24-unit geometry with the glyphs cut out of the square.

| File | Use |
|---|---|
| `logo.svg` | Source artwork, 24-unit grid |
| `icon.png` | 512 px app icon |
| `icon-<n>.png` | 16, 32, 48, 64, 128, 256, and 512 px app icons |
| `icon.ico` | Windows icon with the 16 to 256 px sizes |

Regenerate the PNG and ICO files after changing the geometry in both `logo.svg` and [`scripts/render_icons.py`](scripts/render_icons.py), which needs Pillow:

```sh
python branding/scripts/render_icons.py
```

The mark is a placeholder that has not had design review.
