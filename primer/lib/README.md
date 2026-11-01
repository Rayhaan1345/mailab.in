# Primer library

This folder contains the shared code of How AI Learns: the styles and the Kit helper. The pages are the content. They live in the folders beside this one (`a1-what-is-a-model/`, `b1-one-neuron/`, and so on). A change here reaches all pages after one build.

`assets/kit.css` and `assets/kit.js` are generated from this folder. Do not edit them.

## What is here

| Path | Purpose |
|---|---|
| `css/*.css` | The styles, one file per family. Source of `assets/kit.css`. |
| `kit.js` | The Kit helper: plots, controls, quiz, code blocks, navigation, theme. Source of `assets/kit.js`. |
| `build.py` | Joins the styles, copies `kit.js`, stamps version hashes into the pages. |
| `preview.py` | Makes a noindex copy of the primer with a link back to the live version. |
| `check_styles.mjs` | Compares the computed style of all elements between two copies of the site. |
| `CHANGES-2.3.md` | Step 2.3: which components were unified, which variants were found, and which pages changed. |

A chapter folder has `index.html`, `<slug>.js`, an optional `<slug>.css` and its `data/` folder.

## The style files

`build.py` joins `css/*.css` in filename order. When two rules have the same specificity, the later one wins, so the order below is part of the design.

| File | Contains |
|---|---|
| `00-tokens.css` | The colour palette in light and dark, and the colour-scheme switch. |
| `05-base.css` | Reset, page typography, links, focus rings, `section` and `h2`, reduced motion, print. |
| `10-shell.css` | Page shell: top bar, hero, intro box, "you will need" box, contents, previous and next, footer, theme toggle, references, the primer index. |
| `20-teaching.css` | Teaching blocks: guess first, terms, claim, caption, break box, debug tip, bridge to my project, notes, quick quiz, exercises, Q and A, derivation steps. |
| `30-math-code.css` | Equation boxes and rows, typeset maths with symbol notes, code blocks with output, Python colours, the code walk-through panels of the toolkit chapters. |
| `40-figure.css` | The card around a figure, its controls and readouts, bars and buttons, check boxes, layouts (`split2`, `two`), counters, sliders, token chips, value bars, segment bars. |
| `50-tables.css` | Table styles: `table.tbl`, `.wtab`, `.ptable`, `.sv-table`, and the scroll wrapper `.tscroll`. |
| `60-figdraw.css` | The SVG drawing vocabulary of the generated practice-set figures (axes, grids, labels, series, markers). Use inside `<svg class="fg">`. |
| `70-practice.css` | Practice sets: sticky timer bar, question card, options, results. All of it applies inside `main.pset`. |
| `80-utilities.css` | Utility classes (see below). |

The classes in the selectors of the style files:

**`05-base.css`** (3 classes)  
`.page` `.sc` `.aside`

**`10-shell.css`** (24 classes)  
`.theme-toggle` `.proj` `.hero` `.subtitle` `.byline` `.context` `.pills` `.tldr` `.stats` `.teal` `.coral` `.toc` `.topbar` `.kicker` `.youneed` `.refs` `.chap-nav` `.next` `.part` `.chapter-list` `.num` `.blurb` `.soon` `.soon-link`

**`20-teaching.css`** (33 classes)  
`.cap` `.note` `.plain` `.qa` `.ask` `.terms` `.claim` `.task` `.task-a` `.break` `.recall` `.recall-h` `.quiz` `.ok` `.no` `.was` `.fb` `.debug` `.exercises` `.hint` `.ex-tag` `.used` `.bridge` `.side-h` `.sub-h3` `.datanote` `.dv-step` `.math` `.steps` `.mini-note` `.flagline` `.credit` `.credits`

**`30-math-code.css`** (54 classes)  
`.formula` `.eq` `.eqn` `.t` `.op` `.c-coral` `.c-navy` `.c-teal` `.c-sand` `.math` `.math-notes` `.katex` `.code` `.out` `.kit-copy` `.kit-run` `.tok-k` `.tok-s` `.tok-n` `.tok-c` `.tok-b` `.tok-f` `.eqs` `.eq1` `.status` `.wrap` `.mxrow` `.mxg` `.mxl` `.mx` `.hi` `.mono` `.codeview` `.qcode` `.vb` `.cdetail` `.ln` `.on` `.skip` `.err` `.dim` `.cm` `.vars` `.new` `.gone` `.vbox-title` `.frames` `.fbox` `.cells` `.c` `.cur` `.done` `.kept` `.sel`

**`40-figure.css`** (72 classes)  
`.card` `.bar` `.group` `.btn` `.label` `.pace` `.status` `.dots` `.on` `.replay` `.head` `.out` `.share` `.track` `.fill` `.min` `.plates` `.plate` `.frame` `.flow` `.sliders` `.unit` `.counts` `.hit` `.miss` `.leaves` `.yes` `.redraw` `.plot` `.tall` `.short` `.controls` `.split2` `.two` `.readout` `.katex` `.check` `.chk` `.toks` `.tk` `.c0` `.c1` `.c2` `.unk` `.gone` `.vnone` `.bar-row` `.row-chips` `.four` `.three` `.dg` `.legend-line` `.ring` `.sq` `.fig` `.photo` `.pane-h` `.hintline` `.key` `.dash` `.dot` `.thumbs` `.seg-bar` `.seg` `.c-teal` `.vbar` `.tr` `.fl` `.rs-row` `.rs-tok` `.rs-track` `.pl-sub`

**`50-tables.css`** (17 classes)  
`.sv-table` `.on` `.tbl` `.wtab` `.ptable` `.pick` `.gap` `.words` `.wtab-wrap` `.best` `.cmx` `.grp` `.dg` `.w2` `.tl` `.lft` `.tscroll`

**`60-figdraw.css`** (61 classes)  
`.fg` `.tk` `.nm` `.ax` `.gl` `.gm` `.l1` `.l2` `.d1` `.d2` `.b1` `.vl` `.nd` `.nt` `.ar` `.pt` `.pt2` `.ptx` `.pt2x` `.l3` `.pm` `.ring` `.ring2` `.lb` `.lbc` `.lbm` `.lbx` `.lbt` `.at` `.s1` `.s2` `.s3` `.s4` `.dgn` `.dv` `.bp` `.bn` `.cn` `.lf` `.ld` `.sm` `.vv` `.wt` `.ahp` `.dgr` `.cl` `.ci` `.cp` `.cg` `.cv` `.ix` `.win` `.win2` `.blk` `.bt` `.lb0` `.dbl` `.dm` `.s1d` `.s2d` `.band`

**`70-practice.css`** (43 classes)  
`.pset` `.pbar` `.pb-time` `.over` `.pb-prog` `.pb-jump` `.btn` `.btn-main` `.pq` `.right` `.missed` `.pq-h` `.pq-tag` `.pq-st` `.pq-t` `.tight` `.stm` `.opts` `.opt` `.on` `.katex-display` `.pq-num` `.u` `.revise` `.pq-code` `.wrap` `.pfig` `.fcap` `.pair` `.chips` `.tkc` `.res` `.big` `.sub` `.arow` `.an` `.as` `.meter` `.ar-rev` `.miss` `.acts` `.area-badge` `.wide`

**`80-utilities.css`** (17 classes)  
`.sr` `.hide-s` `.fs-13` `.mt-6` `.mt-8` `.mt-10` `.mt-12` `.mt-14` `.m-b4` `.m-b8` `.mw-36` `.label-block` `.tint-navy` `.tint-coral` `.tint-teal` `.tint-sand` `.tint-ink`


## Where CSS goes

- **Two or more pages need it:** it goes in the library, in the file of its family, as one definition. Search `css/` before you add a rule.
- **One chapter's figure needs it:** it goes in `<slug>/<slug>.css`, next to the page. The page links it right after `kit.css`. The build stamps the link. Prefix the class names with the chapter's prefix (`d3-`, `tl-`, `wk-`).
- **A page has no `<style>` element.** It has no `style=""` attribute that a utility class covers (below). Dynamic values that a script sets, and one-off sizes on SVG parts, may stay inline.
- **Colours are palette variables.** Write `var(--navy)` or `color-mix(in srgb, var(--coral) 20%, transparent)` in CSS, and `Kit.palette()` or `Kit.css("coral")` in scripts. A literal colour is allowed only for a photograph, a white figure plate, the dark terminal block of `.replay`, or a transparent fill. Dark mode comes from the tokens, so a rule needs no dark variant.
- **A page may override a library class** for its own figure, in its `<slug>.css`. The override stays in the chapter when hoisting it would restyle other pages that use the same component. `CHANGES-2.3.md` lists these under "Kept in the chapters".
- **Class names do not change.** Pages keep the names they have.
- **Order and strength.** A rule in `<slug>.css` comes after `kit.css`, so it beats a library rule of the same strength. A library rule that must beat another one in a different file needs a more specific selector. Do not depend on file order for that.
- **Two components with one name.** The library scopes the one it defines so that it cannot reach the other: `.pset .pbar` (the timer bar of the practice sets, F3 has its own `.pbar`), `table.tbl` (C2 uses a `div` called `tbl`), `span.tk` (E4 and the SVG labels of the practice sets have their own `.tk`). Do the same when a name is already taken.
- **Figures that animate are painted from script.** Do not use CSS animations or transitions for figure content.

## How to add a component

- **Check the name.** Search `css/` and the pages for `class="name"` and for `.name`. If another page uses the name for something else, scope your rule as described under "Two components with one name".
- **Pick the family file** by what the thing is: a teaching block in `20-teaching.css`, an equation or code panel in `30-math-code.css`, a part of the figure card in `40-figure.css`, a table in `50-tables.css`, a part of an SVG figure in `60-figdraw.css`, a practice-set part in `70-practice.css`.
- **Write the rule** in a section with a one-line comment above it. Write sizes the way the neighbouring rules do. Put the mobile rule in `@media (max-width: 760px)` directly under the section.
- **Run** `python3 primer/lib/build.py`.
- **Check** with `check_styles.mjs` against the previous copy of the site. Pages that do not use the new class must be identical.
- **If the component is for one chapter only**, put it in that chapter's `<slug>.css` instead.

## Utility classes

They live in `80-utilities.css`. They use `!important` where they replace an inline style, so that they win over a component rule, as the inline style did.

| Class | Effect | Replaces |
|---|---|---|
| `fs-13` | `font-size: 13px` | `style="font-size:13px"` on small SVG text |
| `mt-6`, `mt-8`, `mt-10`, `mt-12`, `mt-14` | margin above, in px | `style="margin-top:12px"` |
| `m-b4`, `m-b8` | no margin above, 4px or 8px below | `style="margin:0 0 8px"` |
| `mw-36` | `max-width: 36em` | a table that should not fill the width |
| `label-block` | `display: block`, 4px below | a label on its own line above a control |
| `tint-navy`, `tint-coral`, `tint-teal`, `tint-sand`, `tint-ink` | sets `--c` to that palette colour | `style="--c: var(--navy)"` on a legend swatch |
| `sr` | visually hidden, still read by screen readers | |
| `hide-s` | hidden below 600px | |

## Build

    python3 primer/lib/build.py            write what is stale
    python3 primer/lib/build.py --check    write no files, exit 1 if something is stale

Run the build after a change in `css/` or `kit.js`, and after you add or edit a page. A second run has no work to do. It uses Python 3 and no other packages.

The build also checks the rule that a page has no `<style>` element. A page may link the library, its own `<slug>.css`, KaTeX and the Google fonts. Any other stylesheet, and any `<style>` element, is reported as a problem and makes `--check` fail.

## Version stamps

A page links the library like this:

    <link rel="stylesheet" href="../assets/kit.css?v=25da757f">
    <link rel="stylesheet" href="b1-one-neuron.css?v=0c1d2e3f">
    <script src="../assets/kit.js?v=e92d596e" defer></script>

The value after `v=` is the first 8 hex digits of the sha1 of the file's content. When the file changes, the value changes, so browsers fetch the new copy. Nobody types a version number. The build stamps a chapter's own `<slug>.js` and `<slug>.css` the same way, and so it does other local scripts and styles next to a page, such as `treelab.js`. Links to other sites are left alone.

## Preview

    python3 primer/lib/preview.py OUTDIR

This runs `build.py --check`, then copies `primer/` to `OUTDIR` without `lib/` and `_template/`. In the copy only, the pages get `noindex`, a canonical link to the live page, and a small pill at the bottom right that reads "Preview v2 · open current version". `OUTDIR` is created, or replaced if it contains an earlier preview. All links in the pages are relative, so the folder can have any name, for example `primer-v2`.

## Checking that styles did not change

    python3 -m http.server 8890    # run in the folder that has the old copy of the site (primer/ inside)
    python3 -m http.server 8891    # run in the folder that has the new copy
    node primer/lib/check_styles.mjs --compare http://127.0.0.1:8890 http://127.0.0.1:8891 --jobs 4

The script needs Node 22 and Google Chrome. For three states of a page (width 1100 light, width 1100 dark, width 390) it waits 9 seconds for the figures, then records a hash of the full computed style of all elements, plus console errors. It prints the page states that differ. It blocks Google web fonts and normalises the ids that Plotly makes up, because both make two runs of one page differ. The other modes are at the top of the script: one page as JSON, snapshot a folder, diff two folders, and `--detail` for property-level differences on one page.

When one rule changes the height of an element, the computed height, `top`, `bottom` and the origins of all its ancestors change too, and so do absolutely placed elements further down the page. To find the element that was restyled, ignore `height`, `width`, `block-size`, `inline-size`, `top`, `bottom`, `left`, `right`, `perspective-origin` and `transform-origin`, and look at what is left.

## Notes

- `_template/` is a starter page. Git ignores it (see `.gitignore`), so it exists only in a working copy. The build stamps it when it is there.
- The practice-set pages and the primer index are generated by scripts outside this repo. Their templates have no style block and write `kit.css` and `kit.js` links with no version. Run `build.py` after generating. Set `PRIMER_OUT=DIR` to make a generator write its page folder into `DIR` for a test run.
- `a5-decision-trees/treelab.js` still writes its own style element at run time (the `.tl-` classes of the tree lab). Its palette colours are `color-mix` of the palette variables.
