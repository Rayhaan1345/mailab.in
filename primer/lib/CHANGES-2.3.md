# Step 2.3: what was unified, and what changed on the pages

This file is for review. Step 2.3 moved the CSS out of the pages. Where one class was defined differently in several pages, the library now has one definition. This file lists all such components: the variants that were found, the definition that was chosen, and the pages whose rendering changed.

Page names are chapter codes: A1 is `a1-what-is-a-model`, D7 is `d7-assistants`, P0 is `p0-practice`, T2 is `t2-algorithms`.

## How the pages were compared

The old copy (the state after step 2.2) and the new copy were loaded in Chrome at 1100 px wide (light and dark) and at 390 px. The full computed style of all elements was hashed and compared. All 47 pages in all three states were compared: 141 page states. The result:

- 12 pages are identical in all three states: B1, B4, C2, C3, Index, PA, PB, PF, T3, T4, T6, T7.
- 35 pages differ, and all differences are listed below.
- There are no console errors in the new copy.

A computed height changes whenever something above it moves, so one change makes dozens of elements differ (the section, the card, the page, absolutely placed elements). To find the cause, the comparison ignores `height`, `width`, `block-size`, `inline-size`, `top`, `bottom`, `left`, `right`, `perspective-origin` and `transform-origin`, and reports the elements that still differ. They are all attributed to a component below. A few differences have no visible effect, for example a flex property on an element that is not a flex container. They are listed at the end, so that all differences are explained.

## Summary

| Component | Variants | Chosen | Pages that change |
|---|---|---|---|
| `bridge` | 3 (17 pages) | teal bar, paper background, 15px text | A5, A6, D6, D7, F1, F2, F3 |
| `used` | 2 (6 pages) | one rule for `p.used` and `div.used` | A1, A2, A3 |
| `eqs`, `eq1` | 5 | row margin, stacked container | D1, D2, D3, D4 |
| `tk`, `toks` (token chips) | 4 chips in 4 pages, and E4 | D1/D3 chip, D5 and D7 keep a modifier | D3, D5, D7 (D3 only in the font stack) |
| `pbar` | 2 (7 pages) | practice bar scoped to `.pset`, F3 keeps its own | none |
| `fg` (drawing vocabulary) | 2 for `.vl` (6 pages) | the longest set, `.vl` with halo | P0 |
| `tbl`, `wtab`, `ptable`, `tscroll` | 2 | majority | A4, A6, T1, T2, T8 |
| `two`, `split2` | 3 gaps (9 pages) | one rule, gap 16px 22px | D3, D4, D5, D6, D7, E1, E2, E3, E4 |
| `check`, `chk` | 4 | one rule, gap 4px | D4, D5, D7, F1, F2, F3 |
| `bar-row` | 2 (7 pages) | no margin | E3, E4 |
| `side-h`, `sub-h3`, `dv-step`, `pl-sub`, `mxrow`, `thumbs`, `seg-bar` | 2 to 4 | the majority value | A2, B2, B3, D2, D3, E4, F1, F2, F3 |
| `.card .bar`, `.card .controls` | page overrides of a library rule | the library rule | A1, A2, A3, B5, B6, B7, B8, B9 |
| practice sets | `pq-code`, `.btn[disabled]` | the newest template | P0, PC, PD |
| palette colours | literals in scripts and styles | the palette variable | A5, C1 (dark), D2, D3, T5 |

The largest change is `bridge` in D6, D7, F1, F2 and F3. Its text goes from 18px to 15px, and the outlined box becomes a teal bar.

## Components

### bridge (`20-teaching.css`)

The "Bridge to my project" box. It is used in 17 pages, in three variants.

| Variant | Pages | Declarations |
|---|---|---|
| A | B5, B7, B8, B9, C3, D5, E1, E2, E3, E4 | `margin: 12px 0 18px; padding: 8px 14px; border-left: 3px solid teal; background: paper; border-radius: 0 6px 6px 0; font-size: 15px; color: ink-2`; paragraphs `margin: 5px 0` |
| B | D6, D7, F1, F2, F3 | `margin: 16px 0; padding: 8px 14px; border: 1px solid rule; border-left: 3px solid sand; background: paper`; no font size, so the text is 18px; paragraphs `margin: 5px 0; color: ink-2` |
| C | A5, A6 | the `.used` box (`margin: 8px 0 18px; padding: 6px 12px; 15px`) with a teal bar and paper background; paragraphs keep the browser margin of 1em |

Chosen: A (10 of 17 pages). The text is 15px, as in the other callouts.

Changes: in D6, D7, F1, F2 and F3 the text goes from 18px to 15px (17px to 15px at 390 px), the outline of the box is removed, the bar goes from sand to teal, and the margin goes from `16px 0` to `12px 0 18px`. Inline maths shrinks with the text (D6 has three formulas in its box). In A5 and A6 the top margin goes from 8px to 12px, the padding from `6px 12px` to `8px 14px`, and the paragraph margins from 15px to 5px.

### used (`20-teaching.css`)

The "Used in" line at the end of a teaching section. The library had `p.used`. A1 to A4 repeated a `.used` rule (`margin: 4px 0 0; font-size: 14.5px; color: muted`) that the more specific library rule always overrode. A5 and A6 used a `div.used` and gave it the library values. The library selector is now `.used`, and the page rules are gone.

Changes: the only live part of A1 to A4's rule was `.used b { font-weight: 600 }`. The library keeps the browser's bold, so bold words in the "Used in" line of A1, A2 and A3 go from 600 to 700.

### eqs, eq1 (`30-math-code.css`)

The equation rows under a figure. Five patterns shared the name `eqs`.

| Pattern | Pages | Declarations |
|---|---|---|
| A: the class on the row itself (`div.eqn.eqs`) | C1, C2, C3, E1, E2, E3, E4 | `margin: 10px 0 0` |
| B: a container that stacks rows | B4, B5, B6, B7 | `margin: 10px 0 0; display: grid; gap: 4px`; rows `margin: 0`; `.eqs .status { margin: 0 0 2px }` (B5, B6) |
| C: a container around one row | D1, D2, D3, D4 | `margin: 10px 0 0`; rows `margin: 6px 0 0` |
| D: a two-column grid | F3 | `display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px` |
| E: `.eq1`, a single row | D5, D6, D7 | `margin: 10px 0 0` |

Chosen: `.eqs, .eq1` give the top margin. A container (`.eqs` that is not an `.eqn`) is a grid with a 4px gap and rows without margin (B). F3 keeps its two columns as an override. C was a container too, with 6px above the row.

Changes: in D1, D2, D3 and D4 the equation row loses its 6px top margin.

### tk, toks: token chips (`40-figure.css`)

The tokenisation chips. Five pages used the name `tk`; one of them (E4) meant something else.

| Variant | Pages | Declarations |
|---|---|---|
| D1/D3 | D1, D3 | a column chip with the id under it: `display: inline-flex; flex-direction: column; align-items: center; padding: 1px 6px 0; line-height 1.35; border: 1px solid transparent` |
| D5 | D5 | `display: inline-flex; padding: 1px 6px; line-height 1.4`; `.toks` with margin `6px 0 10px`; `.gone` with a dashed border and no strike-through |
| D7 | D7 | `align-items: baseline; padding: 1px 5px; border: 1px solid rule; line-height 1.5`; an index `sub` beside the text; `.toks` with `gap: 4px 3px; margin: 6px 0 10px` |
| E4 | E4 | not a chip: a block label inside the token strip (`display: block; font-size: 12.5px; line-height: 24px`) |

Chosen: D1/D3, as `span.tk` (so the SVG labels `text.tk` of the practice sets are not reached). The modifiers `c0`, `c1`, `c2`, `unk`, `gone` come from D1 and D3. D5 keeps its dashed `.gone`. D7 keeps a short override (row direction, baseline, visible border, line height 1.5). E4 resets the chip in its own file, in `.tr-strip .tk`.

Changes: D5 chips are column flex boxes, 1.7px shorter, and its list has 6px below instead of 10px. D7's list has a column gap of 4px instead of 3px and 6px below instead of 10px. The chip font stack is now `ui-monospace, "SF Mono", Menlo, Consolas, monospace` in all four chapters; the face is the same on a Mac. E4 is unchanged.

### pbar and the other practice-set names (`70-practice.css`)

The sticky timer bar of the practice sets (six identical copies) shares its name with F3's probability bar, a grid of label, track and number. The practice rules are scoped: `.pset .pbar`, `.pset .opts`, `.pset .res`, `.pset [hidden]`. F3 keeps its `.pbar` rules. `.opts` is also used by D1 and F3, and `.res` by C1 and D2, for other things.

Changes: none.

### fg: the drawing vocabulary (`60-figdraw.css`)

The six practice pages had the same `.fg` rules, pasted in. The sets grow from P0 (15 `.fg` rules) to PF (61), so the rules were unioned. Only `.fg .vl` (value labels) differed: PD and PF draw a halo behind the label (`paint-order: stroke; stroke: var(--bg); stroke-width: 3.5px`), the other four do not.

Chosen: the halo (PD, PF, the newest template).

Changes: P0 draws the halo too (white in light mode, the dark page colour in dark mode). PA, PB and PC define the label, but their figures show no value label at load.

### Practice sets: other differences (`70-practice.css`)

- `.btn[disabled]`: the practice pages used `opacity: 0.45`, the library `0.4`. The library value is kept.
- `.pq pre.pq-code`: PF alone made code in a question 12px at 390 px. All practice sets now do.
- `.tscroll`: see Tables.

Changes: at 390 px the code blocks in questions are 12px instead of 13.5px in P0, PC and PD. The disabled button change needs a disabled button on screen.

### Tables (`50-tables.css`)

Four table styles grew separately. They share one rule (`border-collapse`, tabular figures). They keep their own sizes, because the row density differs.

- `.tbl` (A4, A6, F1, F2, F3). F1, F2 and F3 set `vertical-align: top`, A4 and A6 did not. F1 and F2 keep the first column of `.words` tables on one line, F3 does not. Chosen: `vertical-align: top` and no wrapping. F3 keeps `white-space: normal` as an override. The selector is `table.tbl`, because C2 uses a `div` called `tbl` as a scroll wrapper.
- `.wtab` (A1, A3): identical, moved as one.
- `.ptable` (the six practice sets): identical, moved as one. Its two shared declarations come from the base rule.
- `.sv-table` stays in the library. D4's overrides stay in D4.
- `.tscroll` (the six practice sets, T1, T2, T8). The practice sets use `margin: 4px 0 8px`, the toolkit chapters `10px 0 0`. Chosen: the practice value (six pages).

Changes: A4 and A6: table cells align at the top (98 cells in A4, 52 in A6, all three states). It shows in rows where a cell wraps. T1, T2 and T8: the scroll wrapper has 4px above and 8px below instead of 10px above.

### two, split2: two-column layout (`40-figure.css`)

`.split2` was in the library. `.two` was defined in D3, D4, D5, E1, E2, E3 and E4 with the same columns, and D6 and D7 repeated `.split2` with a different gap.

| Variant | Pages | Gap |
|---|---|---|
| library | the 19 pages that use `split2` | `16px 22px` |
| D | D3, D4, D5, D6, D7 | `12px 22px` (D3 also `1.25fr 1fr`) |
| E | E1, E2, E3, E4 | `14px 22px` |

Chosen: one rule for `.split2, .two` with the library gap. D3 keeps its `1.25fr` column as an override. F3 uses `two` as a modifier on `.counts` and `.step-body`; it resets `gap` and `align-items` for `.counts.two`.

Changes: the row gap between stacked panels goes from 12px (D3 to D7) or 14px (E1 to E4) to 16px.

### check, chk (`40-figure.css`)

A label with a checkbox. `.check` (C1, C2, C3, E1, E2, E3, E4) had `gap: 4px; font-size: 14.5px; cursor: pointer`. `.chk` had `gap: 6px` with no pointer (D4, D5, F1, F2, F3), and in D7 `gap: 4px; font-size: 14px; align-items: flex-start; max-width: 100%`.

Chosen: one rule for `.check, .chk` (`gap: 4px`, 14.5px, pointer). D7 keeps `align-items: flex-start; max-width: 100%` for `label.chk`.

Changes: F1, F2, F3: gap 6px to 4px, and a pointer cursor. D4, D5: pointer cursor (their labels in a bar keep the 6px gap from `.card .bar label`). D7: pointer cursor and 14.5px text.

### Identical copies and small drifts (`20-teaching.css`, `30-math-code.css`, `40-figure.css`)

These had identical copies in several pages. They moved without change: `row-chips` (D1 to D7), `pane-h` (C1, C2, C3, E1, E2, E3, E4), `hintline` (C1, C2, C3, E2, E3), `key` (C2, C3, E2, E3), `counts.three` and `counts.four`, `canvas.fig` (C1, C2, C3), `btn-main` (the six practice sets), `credit` and `details.credits` (C1, C3, E1 to E4; F3 keeps `.photo-box .credit`), `datanote` (A5, A6), `flagline` (D1 to D3), `mini-note` (A4, A6, F1, F2, F3), `ol.steps` (B2, B3), and the code walk-through panels of T1 and T2 (`cells`, `codeview`, `vb`, `vars`, `frames`, `fbox`).

Variants that differed:

| Component | Variants | Chosen | Changes |
|---|---|---|---|
| `bar-row` | C1, C2, C3, E1, E2: no margin. E3, E4: `margin: 0 0 8px` | no margin | E3, E4: 8px less below the row |
| `legend-line` | A4, F1, F2, F3: a flex row. T4: a paragraph with `margin: 0 0 8px; color: muted` | the flex row | T4 keeps a short override (`display: block`, its margin and colour) |
| `side-h` | D4, D5, D7: `margin-top: 6px`. D2, D3: 8px | 6px | D2, D3: 2px less above |
| `sub-h3` | A4 26px/8px; B3 30px/4px and 20px text; F2 22px/8px; F3 24px/8px; F1 had the values inline | A4 (the same as `.exercises h3`) | B3: top 30px to 26px, bottom 4px to 8px, text 20px to 19px; F1: 22px to 26px (the inline style became the class); F2: 22px to 26px; F3: 24px to 26px |
| `dv-step` | A3, B3: `margin-bottom: 10px`. A2: 12px | 10px | A2: 2px less below |
| `pl-sub` | B2: `14px 0 4px`. B3: `10px 0 2px` | B2 | B3: 4px more above, 2px more below |
| `mxrow`, `mxrow .op`, `table.mx td` | B1: margin-top 14px, operator 20px, cells `padding: 2px 8px`. B2: 12px, 19px, `padding: 2px 5px; min-width: 2.3em` | B1 | B2: 2px more above the row, operator 1px larger. B2 keeps its cell override. |
| `thumbs` | E2: `margin: 0 0 10px`. F3: no margin | E2 | F3: 10px more below the thumbnails |
| `seg-bar` | E3: `margin: 8px 0 0`. E4: 6px | E3 | E4: 2px more above |
| `vbar` | A4: label column 112px, `.tr` positioned. F1: 150px | A4 | F1 keeps its 150px columns as an override. F1's `.tr` is positioned (no visible effect). |
| `rs-row`, `rs-track i`, `rs-row b` | D5, D7: 8.4em label column. E2: 13.5em, 14.5px, navy bar, ink number | D5/D7 | E2 keeps its columns, size and colours as an override |
| `eqn .op.wrap` | E1, E2: `white-space: normal`. C3: also `max-width: 22em; font-size: 15px; color: ink-2` | E1/E2 | C3 keeps the extra declarations |

### Library classes that pages overrode

`.card .bar` (A1, A2, A3) and `.card .controls` (B5 to B9) were overridden by pages with spacing that the other pages do not use. The pages now use the library rules. Changes: A1, A2, A3: the bar's column gap goes from 10px to 18px and the space below from 10px to 12px. B5, B6, B7, B8, B9: the space above the controls goes from 10px to 4px.

## Not unified: one name, different components

These classes are defined in two or three pages, but the pages mean different things by the name. Merging them would change layouts, so these pages keep their own rules in their own `<slug>.css` files. A later step can rename them.

| Class | Pages | What they are |
|---|---|---|
| `badge`, `badge.script` | D7, F3 | D7: a pill above a figure. F3: a small status label in a table of steps. |
| `strip` | C3, T2 | C3: a grid of 8 image tiles. T2: a flex row of array cells. |
| `qline` | D7, T8 | D7: a highlighted question line. T8: an italic line of prose. |
| `verdict` | B8, D2, F3 | B8: bold text. D2: a bar with a sand edge. F3: a boxed result. |
| `ly-row` | B1, B4 | B1: a grid row of a layer table. B4: a flex row of controls. |
| `pair` | D7, E1, E2 | D7: a two-column grid of buttons. E1 and E2: a row of canvases that wraps (centred in E1). The practice sets use `.pfig .pair`, a grid of two figures. |
| `opts` | D1, F3 | D1: a row of selects. F3: a grid of answer buttons. The practice sets use `.pset .opts`. |

## Kept in the chapters: page tweaks of library classes

A page may restyle a library component for its own figure. Moving such a tweak into the library would restyle the other pages that have the same component, so it stays in the chapter's CSS file. The last column counts those other pages. For the rules that the library already has, the page rule is an override and the library rule is the default.

| Selector | Pages | Other pages with matching elements |
|---|---|---|
| `.card .status` | A1, A2, A3 | 34 |
| `.card .bar .label` | A4, A5, F1, F2, F3, T3, T4, T5, T6, T7 | 20 |
| `.card .group` | A4, F1, F2 | 43 |
| `.card .eqn` | A5, A6, T3, T4, T5 | 34 |
| `.group` at `@media (max-width: 760px)` | C1, C2, C3, E1, E2 | 41 |
| `.card .bar label` | D2, D3, D4, D5 | 8 |
| `.card .bar select, .card .controls select` | D2, F1 | 16 |
| `.card .bar select` | D3, D4, D5 | 3 |
| `.card input[type=checkbox], .card .bar input[type=range]` | D4, D5, D6, D7, F1, F2, F3 | 11 |
| `.card .bar label:has(input[type=range])` | D5, D6, D7 | 2 |
| `.card .bar label input[type=range]` | D5, D6, D7 | 2 |
| `.split2 > div` | D6, D7 | 17 |
| `.card input[type=range]` | E1, E2, E3, E4 | 35 |
| `.card canvas` | E1, E2, E3, E4 | 12 |
| `.card select` | E2, E3 | 18 |
| `svg .drag` | T4, T5 | 36 |
| `svg .drag:focus` | T4, T5 | 36 |
| `svg .drag:focus-visible` | T4, T5 | 36 |
| `.card svg` | A4, T7 | the rule is also in the library |
| `.card .controls label` | B4, E2, E3, E4 | the rule is also in the library |
| `.card .controls select` | B4, E2, E3 | the rule is also in the library |
| `.eqn .t` | D2, D3, D4, D5, D6, D7 | the rule is also in the library |

## Other changes from moving the CSS

- **Leftovers removed.** The commented block at the end of `05-base.css` is gone. The only names in it that occur in the pages are `seg` (E3, E4) and `fig` (`canvas.fig` in C1, C2, C3), and `.fig` set only `margin: 0`, which a canvas has anyway. E3 and E4 now define `.seg-bar .seg` completely. Their `.seg` spans lose a `flex-wrap` value that had no effect on an element that is not a flex container.
- **`.actions` in the print rule.** `10-shell.css` hid `.actions` and `.theme-toggle` when printing. No page uses `.actions`, so the rule names only `.theme-toggle`.
- **Order of rules.** The library files are read in filename order and then the chapter file. The rules moved from a page keep their order inside their block. The media queries stay beside the rules they belong to.
- **Strings in CSS.** The content string of T2 `.stackbox .it.top::after` has two spaces before the arrow. The move keeps both.

## Palette colours

Colours that were written as literal values now come from the palette, so a palette change reaches them.

| Where | Was | Now |
|---|---|---|
| D2 (2 Plotly lines), D3 (3 Plotly lines) | `#c2563a`, `#3b4f82`, `#8f98a2` | `Kit.css("coral")`, `Kit.css("navy")`, `Kit.css("muted")` |
| C1 (3 fills), C2 (2 fills), E1 (1 fill) | `rgba(194,86,58,a)`, `rgba(15,154,128,a)` | the palette colour with the same alpha, read from `Kit.palette()` |
| `a5-decision-trees/treelab.js` (injected styles) | `rgba(15,154,128,a)`, `rgba(168,149,124,a)` | `color-mix(in srgb, var(--teal) a%, transparent)`, same for sand |
| T1 (`.bc.x`, `.cgrid button.cc.bad`), T1 and T2 (`.codeview .ln.err`), B8 (`.ck-res.fail .ck-badge`), T5 (`table.mat td.cs`) | `rgba(194,86,58,a)` | `color-mix(in srgb, var(--coral) a%, transparent)` |
| F3 `.tile.lit` | `#fff`, and `#0f1317` in dark mode | `var(--bg)`, which is both values |

In light mode these render as before. In dark mode they now use the dark palette colour. At load this shows in T5 (`td.cs`), A5 (the `tl-yes` and `tl-no` boxes), C1 in dark mode (the coral fill of the size grids) and the D2 and D3 lines, which are now the theme's muted or coral line colour in both modes. The others appear after interaction.

### Colours left on purpose

| File | Colour | Reason |
|---|---|---|
| `lib/css/40-figure.css` `.replay` | `#0d1216`, `#d9e1e8`, `#8a96a3` | the dark terminal block, dark in both themes |
| `00-tokens.css` `--shadow`, `70-practice.css` `.pset .pbar` | `rgba(0, 0, 0, 0.04)`, `rgba(0, 0, 0, 0.07)` | shadows, not palette colours |
| E4 `#fl-t-img` | `#000000` | black behind a photograph |
| F2 `#qz-grid .dg canvas`, F3 `.digit canvas`, `.photo-box canvas, img`, `.thumbs button img` | `#ffffff` | white plates under images of digits and photographs |
| B9 `.stage .patch` | `rgb(124,116,104)` | the colour of a patch in the photograph |
| C1 `c1-convolution.js` line 156 | `#fff` and `#17191c` | label colour chosen by the luminance of the cell behind it |
| C1 229, C2 267 to 283, T1 362 to 363 | `#fff`, `#000` | white strokes and black fields of drawn images and masks |
| C3 114, E1 325, E2 398 | `#fff` | white ring around a marker over an image |
| E1 177 to 185, 247, 361, 402, 441, 533; E4 120; E3 242; B9 40 | `rgba(255,255,255,a)`, `rgba(0,0,0,a)`, `rgba(128,128,128,a)` | grid lines and dimming over photographs |
| E1 185, E2 203, F3 282 | `#fff` | text on a dark overlay, on a heat cell, on a node circle |
| A1, A2, A3, B4, B5, B6, C1, C2, C3, E1, E2, E3, E4, T6, T7 | `#888888` in `hexRGB` | the grey that the helper returns when a colour variable is empty |
| B9 `rgb()` | `#c2563a`, `[194,86,58]` | the same kind of fallback, for coral |
| A1, A2, A3, B2, D2, D3, D4, D5, E1, E3, F1, F2, T8, `treelab.js`, `kit.js` | `rgba(0,0,0,0)` | a transparent fill or background, not a palette colour |
| C1, C3, E2, E3, E4, F3 | `rgb(...)` built from image pixels | photograph colours |

## Inline styles

| Class | Replaces | Uses | Pages |
|---|---|---|---|
| `mt-12` | `style="margin-top:12px"` | 22 | A1, A4, B2, B3, E1, E2, F1 |
| `mt-10` | `style="margin-top:10px"` | 18 | A2, A3, A4, E1, E3, E4, T6 |
| `m-b8` | `style="margin:0 0 8px"` | 3 | A2, A3, A4 |
| `tint-navy` | `style="--c:var(--navy)"` | 8 | A4, F1 |
| `tint-sand` | `style="--c:var(--sand)"` | 3 | A4, F1 |
| `tint-coral` | `style="--c:var(--coral)"` | 7 | A4, F1 |
| `tint-ink` | `style="--c:var(--ink)"` | 4 | A4 |
| `tint-teal` | `style="--c:var(--teal)"` | 3 | A4, F1 |
| `mw-36` | `style="max-width:36em"` | 2 | A6 |
| `mt-14` | `style="margin-top:14px"` | 1 | C2 |
| `m-b4` | `style="margin:0 0 4px"` | 3 | D1 |
| `label-block` | `style="display:block;margin-bottom:4px"` | 3 | D1 |
| `mt-8` | `style="margin-top:8px"` | 2 | E1, E2 |
| `mt-6` | `style="margin-top:6px"` | 1 | E3 |
| `sub-h3` | `style="margin:22px 0 8px;font-size:19px;font-weight:600"` | 2 | F1 |
| `fs-13` | `style="font-size:13px"` | 52 | PC |

The `h3` of F1 had the values of `.sub-h3` inline except for the top margin (22px). It now has the class, so its top margin is 26px (see `sub-h3`). The 52 `font-size:13px` attributes are on SVG text in PC; the generator now writes the class.

Twenty-six inline styles are left. They are values a script computes (stroke widths of SVG parts in PA, PB, PD), elements a script shows or hides (`display:none` in A4 and B8), sizes of plot areas (D2) and one-offs (a width in A4 and F1, and single margins and fonts in B2, C1, C2, D1, P0, PB, PC and T7).

## Differences with no visible effect

| Page | Element | Property | Cause |
|---|---|---|---|
| A3 | `span.bar` in `.sm-table` | `column-gap` 10px to 18px | the `.card .bar` override is gone; the spans are `inline-block` |
| D3, D5, D7 | `span.tk`, `sub` | `font-family` lists SF Mono and Consolas | one font stack for all chips; the first entry already wins on macOS |
| D1, D2, D3, D4 | `.eqn` in `.eqs` | `min-height`, `min-width` `0` to `auto` | the container is a grid, and grid items have an automatic minimum |
| D2 | `.co-grid`, `.mp-side` | `grid-template-rows` | the tracks follow the 2px smaller `side-h` margin |
| D6 | KaTeX spans in the bridge | `font-size` 19.44px to 16.2px | KaTeX is 1.08em of the bridge text, which went from 18px to 15px |
| E3, E4 | `span.seg` | `flex-wrap` | see "Other changes" |
| F1 | `.vbar .tr` | `position: relative` | A4's rule is canonical |
| F3 | `#tr-controls` | `grid-template-rows` | the thumbnail row is 10px taller (`thumbs`) |
| A5 | `.tl-yes`, `.tl-no` | colour written as `color(srgb ...)` | the same colour, now built with `color-mix` from the palette |
