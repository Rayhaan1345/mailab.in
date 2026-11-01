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

## The style files

`build.py` joins `css/*.css` in filename order. When two rules have the same specificity, the later one wins, so the order below is part of the design.

| File | Contains |
|---|---|
| `00-tokens.css` | The colour palette in light and dark, and the colour-scheme switch. |
| `05-base.css` | Reset, page typography, links, focus rings, `section` and `h2`, reduced motion, print. It ends with a block of personal-site leftovers (see below). |
| `10-shell.css` | Page shell: top bar, hero, intro box, "you will need" box, contents, previous and next, footer, theme toggle, references, the primer index. |
| `20-teaching.css` | Teaching blocks: guess first, terms, claim, caption, break box, debug tip, notes, quick quiz, exercises, Q and A. |
| `30-math-code.css` | Equation boxes, typeset maths with symbol notes, code blocks with output, Python colours. |
| `40-figure.css` | The card around a figure, its controls and readouts, bars and buttons, layouts, counters, sliders. |
| `50-tables.css` | Table styles shared by chapters. |
| `60-figdraw.css` | SVG drawing vocabulary (axes, grids, labels, series, markers). Empty for now. |
| `70-practice.css` | Practice sets: timer bar, question card, options, results. Empty for now. |

The last two files are empty on purpose. The pages still contain their own copies of those rules. Moving the copies here is the next step.

## Build

    python3 primer/lib/build.py            write what is stale
    python3 primer/lib/build.py --check    write no files, exit 1 if something is stale

Run the build after a change in `css/` or `kit.js`, and after you add or edit a page. A second run has no work to do. It uses Python 3 and no other packages.

## Version stamps

A page links the library like this:

    <link rel="stylesheet" href="../assets/kit.css?v=25da757f">
    <script src="../assets/kit.js?v=e92d596e" defer></script>

The value after `v=` is the first 8 hex digits of the sha1 of the file's content. When the file changes, the value changes, so browsers fetch the new copy. Nobody types a version number. The build stamps a chapter's own `<slug>.js` and `<slug>.css` the same way, and so it does other local scripts and styles next to a page, such as `treelab.js`. Links to other sites are left alone.

## Where a style goes

- A class that two or more pages need goes in the library, in the file of its family. Keep one definition per component. Search `css/` before you add a rule.
- A class that only one figure uses stays with the chapter: in the page's `<style>` block, or in a `<slug>.css` file beside the page. The build stamps that file too.
- Class names do not change. Pages keep the names they have.
- Order matters. If a rule must override one in another file, give it a more specific selector. Do not depend on file order for that.
- Figures that animate are painted from script. Do not use CSS animations or transitions for figure content.

## Leftovers

The primer started from the stylesheets of the personal site. Rules for classes that appear nowhere in the pages or scripts were removed. Rules for classes that occur only as plain words somewhere were kept. They are at the end of `05-base.css` under their own comment, and no page markup is known to use them. They can go once that is confirmed.

## Preview

    python3 primer/lib/preview.py OUTDIR

This runs `build.py --check`, then copies `primer/` to `OUTDIR` without `lib/` and `_template/`. In the copy only, the pages get `noindex`, a canonical link to the live page, and a small pill at the bottom right that reads "Preview v2 · open current version". `OUTDIR` is created, or replaced if it contains an earlier preview. All links in the pages are relative, so the folder can have any name, for example `primer-v2`.

## Checking that styles did not change

    python3 -m http.server 8890    # run in the folder that has the old copy of the site (primer/ inside)
    python3 -m http.server 8891    # run in the folder that has the new copy
    node primer/lib/check_styles.mjs --compare http://127.0.0.1:8890 http://127.0.0.1:8891 --jobs 4

The script needs Node 22 and Google Chrome. For three states of a page (width 1100 light, width 1100 dark, width 390) it waits 9 seconds for the figures, then records a hash of the full computed style of all elements, plus console errors. It prints the page states that differ. It blocks Google web fonts and normalises the ids that Plotly makes up, because both make two runs of one page differ. The other modes are at the top of the script: one page as JSON, snapshot a folder, diff two folders, and `--detail` for property-level differences on one page.

## Notes

- `_template/` is a starter page. Git ignores it (see `.gitignore`), so it exists only in a working copy. The build stamps it when it is there.
- The practice-set pages and the primer index are generated by scripts outside this repo. Their templates write `kit.css` and `kit.js` links with no version. Run `build.py` after generating.
