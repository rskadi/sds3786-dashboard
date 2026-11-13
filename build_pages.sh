#!/bin/sh
# Construit la version statique (GitHub Pages) dans docs/ : Python tourne dans le navigateur via Pyodide.
# À relancer après chaque modification de app.py ou des données.
set -e
rm -rf docs
.venv/bin/panel convert app.py --to pyodide-worker --out docs --title "Inflation & chômage" \
  --requirements pandas numpy "plotly>=5.20,<6"
mv docs/app.html docs/index.html
mkdir -p docs/data/clean && cp data/clean/*.csv docs/data/clean/
touch docs/.nojekyll
echo "OK : docs/ prêt. Test local : python3 -m http.server 8765 --directory docs"
