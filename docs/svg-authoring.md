# SVG authoring (publishable figures and data viz)

Conventions for producing standalone `.svg` figures meant for the web (GitHub Pages, blogs, docs). Follow these unless told otherwise.

## File setup
- Emit a real standalone file: `<?xml version="1.0" encoding="UTF-8"?>` then `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H" width="W" height="H">`. The `xmlns` is required or the file will not render as a standalone image.
- Set `font-family` once on the root `<svg>` so all text inherits it.
- Add `role="img"`, a `<title>` (one-line description), and `aria-labelledby` pointing at it. Add a longer `<desc>` for complex figures.
- Keep it script-free and with no external fetches. Self-contained SVG embeds and ports everywhere.

## Transparency
- Never add a full-canvas background `<rect>` unless asked. Leave the canvas transparent so the figure sits on whatever the page background is.

## Light and dark theme adaptation
- Put a `<style>` block in the SVG and adapt text and line colors with `@media (prefers-color-scheme: dark)`. Drive it through classes (`.t` primary text, `.m` muted text, `.axis` lines, `.track` bar backgrounds).
- Keep data and brand colors as fixed hex fills. Pick mid-tones (roughly the 400 to 600 range of a ramp) that read on both light and dark backgrounds.
- Use off-black and off-white for text, not pure `#000` / `#fff`. Good defaults: `#2c2c2a` light, `#e8e6dd` dark.
- Caveat 1: `prefers-color-scheme` follows the reader's OS setting, not a site theme toggle. If the target site toggles theme with a class or attribute (for example `[data-theme="dark"]`), target that selector instead of, or in addition to, the media query.
- Caveat 2: this works when the SVG is inline, or embedded via `<img>` or `<object>`, on a normal site such as GitHub Pages. It does NOT work in a github.com README, because GitHub sanitizes embedded `<style>` and scripts and proxies the image. For READMEs, ship a static figure or a 2x PNG.

## Fonts
- Never assume a specific font is installed on the reader's machine. Use a system stack: `system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`.
- SVG `<text>` does not wrap. Break lines manually with `<tspan x="..." dy="1.2em">`. If a label needs wrapping, it is usually too long, so shorten it.
- Estimate text width at about 8px per character for 14px sans (more for bold, numerals, and symbols) and size boxes accordingly so text does not overflow.

## Coordinates and layout
- Set viewBox height to the bottom edge of the lowest element plus about 20px. No wasted space, no clipping.
- No negative coordinates. Keep all content inside the viewBox.
- `text-anchor="end"` extends the text leftward from its x. Verify long right-aligned labels do not run past x=0.
- Use thin strokes (0.5px to 1px). They read more refined than 2px.

## Color
- Color should encode meaning, not sequence. Group by category, use two or three colors, and reserve gray for neutral or structural elements. Do not cycle through a rainbow.
- Text on a colored fill: use a dark shade of the same hue, or white only on mid-dark fills. Avoid placing text on light fills.

## Generate, do not hand-place
- For anything data-driven (bars scaled to values, dot or strip plots, repeated cells, gridlines, axes), write a small Python or JS generator that computes the coordinates. Hand-placed coordinates drift and overlap.
- Round every number that reaches the screen. Float math leaks artifacts like `0.30000000000000004`.

## Verify before shipping
- Rasterize to PNG and look at it. Coordinate bugs (overlaps, clipping, off-canvas text) are invisible in the XML. `python3 -c "import cairosvg; cairosvg.svg2png(url='f.svg', write_to='f.png', background_color='white')"`, or `rsvg-convert`/`inkscape` if installed. Render once on white and once on a dark fill to confirm both themes.

## Embedding (responsive)
- `<img src="fig.svg" alt="..." style="max-width:100%;height:auto">`, or set `width="100%"` on the root `<svg>`.
- Provide a 2x PNG fallback for surfaces that do not render SVG: social and Open Graph cards, some chat and preview panes, email.

## Canonical starter template

```svg
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 H" width="720" height="H"
     role="img" aria-labelledby="ttl"
     font-family="system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">
  <title id="ttl">One-line description for screen readers</title>
  <style>
    .t{fill:#2c2c2a}.m{fill:#6b6a64}.axis{stroke:#d3d1c7}.track{fill:#eceae2}
    @media (prefers-color-scheme:dark){
      .t{fill:#e8e6dd}.m{fill:#a8a69c}.axis{stroke:#45453f}.track{fill:#2e2e2b}
    }
  </style>
  <!-- data shapes: fixed hex fills that read on both themes -->
  <!-- text: class "t" (primary) or "m" (muted); lines: class "axis"; bar backgrounds: class "track" -->
</svg>
```

## Reusable palette (reads on light and dark)
- Teal `#1D9E75`, coral `#D85A30`, blue `#378ADD`, strong blue `#185FA5`, amber `#EF9F27`, neutral gray `#888780`, light gray `#C7C5BC`.
