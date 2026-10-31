# mailab.in

Source of the Mumbai AI Lab website, served at [mailab.in](https://mailab.in).

The site is plain HTML, CSS and JavaScript with no build step.

- `index.html` holds the page. New updates go at the top of the `Updates` list.
- `assets/mailab.css` holds the styles. Colours come from the Socratic Tutor icon, and the type is Anek Latin and Anek Devanagari by Ek Type.
- `assets/mailab.js` runs the hint ladder demonstration and builds the email link when it is clicked.
- `socratic-tutor/index.html` is the report on how the plugin works. `assets/report.css` adds its styles on top of `mailab.css`.
- `assets/socratic.js` runs the report's figures: the Meno scene, the study charts, the file viewer, the skill-loading demonstration, the tutor simulator and the subject notes.
- `assets/img/email.png` shows the contact address as an image, drawn in Chalkduster at 96 px on a transparent background, so scrapers do not read it.
- `CNAME` names the custom domain for GitHub Pages.

To preview locally, run `python3 -m http.server` in this folder and open http://localhost:8000.
