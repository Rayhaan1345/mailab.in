# mailab.in

Source of the Mumbai AI Lab website, served at [mailab.in](https://mailab.in).

The site is plain HTML, CSS and JavaScript. The home page and the Socratic Tutor report are written by hand and have no build step. The primer in `primer/` is built from source files by a script (see below).

- `index.html` holds the page. New updates go at the top of the `Updates` list.
- `assets/mailab.css` holds the styles. Colours come from the Socratic Tutor icon, and the type is Anek Latin and Anek Devanagari by Ek Type.
- `assets/mailab.js` runs the hint ladder demonstration and builds the email link when it is clicked.
- `socratic-tutor/index.html` is the report on how the plugin works. `assets/report.css` adds its styles on top of `mailab.css`.
- `assets/socratic.js` runs the report's figures: the Meno scene, the study charts, the file viewer, the skill-loading demonstration, the tutor simulator and the subject notes.
- `assets/img/email.png` shows the contact address as an image, drawn in Chalkduster at 96 px on a transparent background, so scrapers do not read it.
- `CNAME` names the custom domain for GitHub Pages.
- `primer/` is the interactive primer How AI Learns. A chapter folder (`primer/a1-what-is-a-model/`) has `page.html`, the source you edit (front matter and content), and `index.html`, which the build writes from it. Do not edit `index.html`, `primer/assets/kit.css` or `primer/assets/kit.js`: the next build overwrites them.
- `primer/lib/` has the shared parts: the page template, the styles in `css/`, `kit.js`, the build script, and `snippets/` for text that repeats across pages (facts about the projects, licence and credit lines). The chapter list is `primer/chapters.json`. `primer/lib/README.md` has the details.

To build the primer, run `python3 primer/lib/build.py` (Python 3, no other packages) after you edit a `page.html`, a file in `primer/lib/` or `chapters.json`. Run `python3 primer/lib/build.py --check` to see whether a file is out of date. It writes no files and exits with 1 when the build is stale.

To preview locally, run `python3 -m http.server` in this folder and open http://localhost:8000.
