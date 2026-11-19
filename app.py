"""Tableau de bord : inflation et chômage dans le monde (1991-2021).

Lancer :  panel serve app.py --show
(Exécuter d'abord `python prepare_data.py` pour créer data/clean/.)

Organisation : données -> thème (couleurs, seuils) -> widgets -> calculs -> cartes KPI et phrases -> figures -> page.
"""
import io
import sys

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
from plotly.subplots import make_subplots

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

# ================================================================ données
def read_data(name):
    """Fichier local avec `panel serve`. Dans le navigateur (GitHub Pages), c'est le navigateur qui télécharge
    le fichier : pandas ne reçoit que le texte. (Si pandas lisait l'URL lui-même, il verrait l'en-tête
    « gzip » du serveur et tenterait de décompresser un texte déjà décompressé -> BadGzipFile.)"""
    if "pyodide" in sys.modules:
        from urllib.parse import urljoin
        import js
        from pyodide.http import open_url
        return pd.read_csv(open_url(urljoin(str(js.location.href), f"data/clean/{name}")))
    return pd.read_csv(f"data/clean/{name}")


PANEL = read_data("panel.csv")
PROFILES = read_data("countries.csv")
PANEL = PANEL.merge(PROFILES[["iso3", "profile"]], on="iso3")
NAMES = PANEL.drop_duplicates("iso3").set_index("iso3")["country"].sort_values()

MEASURES = {"Globale": "inflation", "Alimentaire": "food_inflation", "Énergie": "energy_inflation"}
SHOCKS = {1997: "Crise asiatique", 2008: "Crise financière", 2020: "COVID-19"}
PERIODS = [(1991, 2000), (2001, 2010), (2011, 2021)]
HYPER = 50   # hyperinflation toujours masquée (valeurs extrêmes, ex. Venezuela 65 374 %)
HIGH_INFL = 10  # seuil fixe des couleurs « inflation très élevée » (les alertes ont leurs propres curseurs)
TARGET = 2   # cible d'inflation des banques centrales

# Profils standardisés (mêmes attributs que le KMeans) pour trouver les pays les plus semblables
_F = PROFILES.set_index("iso3")[["unemp_median", "unemp_std", "infl_median", "infl_std", "phillips_r"]].dropna()
_F[["infl_median", "infl_std"]] = np.sign(_F[["infl_median", "infl_std"]]) * np.log1p(_F[["infl_median", "infl_std"]].abs())
PROFILE_Z = (_F - _F.mean()) / _F.std()

# ================================================================ thème sombre
FONT = "Inter, system-ui, sans-serif"
BG, SURFACE, SURFACE2 = "#111110", "#1a1a19", "#232321"     # fond de page, cartes, survols
BORDER, GRID = "#2c2c2a", "#262624"
TEXT, TEXT2, TEXT3 = "#f0efec", "#c3c2b7", "#8d8c85"         # principal, secondaire, discret
# Couleurs de séries (identité), versions pour fond sombre ; une couleur fixe par catégorie
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#3987e5", "#d95926", "#199e70", "#c98500", "#e66767", "#7d7c76"
ACCENT = BLUE
SERIES_COLORS = {"Globale": BLUE, "Alimentaire": AQUA, "Énergie": ORANGE}
PROFILE_COLORS = dict(zip(PROFILES.profile[PROFILES.profile != "Données insuffisantes"].value_counts().index,
                          [BLUE, ORANGE, AQUA, YELLOW]))  # le plus grand groupe -> 1re couleur
PROFILE_COLORS["Données insuffisantes"] = GRAY
INCOME_COLORS = {"Revenu élevé": BLUE, "Revenu intermédiaire supérieur": ORANGE,
                 "Revenu intermédiaire inférieur": AQUA, "Faible revenu": YELLOW, "Non classé": GRAY}
DIVERGING = [[0, BLUE], [0.5, "#383835"], [1, RED]]
PATH_COLORS = [BLUE, ORANGE, AQUA, YELLOW, "#d55181", "#9085e9", "#2f9e44", RED]  # une couleur fixe par pays choisi  # r : bleu = Phillips (r < 0), rouge = sens inverse

# Couleurs d'état — toujours accompagnées d'une icône et d'un libellé
STATUS = {  # niveau: (couleur, icône)
    "good": ("#0ca30c", "✓"),
    "warn": ("#fab219", "!"),
    "bad": ("#e5484d", "⚠"),
    "cold": ("#86b6ef", "↓"),  # déflation
    "neutral": (TEXT3, ""),
    "info": (BLUE, ""),        # r : ni bon ni mauvais
}


def unemp_level(v):
    if pd.isna(v):
        return "neutral", "n/d"
    if v < 5:
        return "good", "Faible (< 5 %)"
    if v < 10:
        return "warn", "Modéré (5–10 %)"
    return "bad", "Élevé (≥ 10 %)"


def infl_level(v, thr):
    if pd.isna(v):
        return "neutral", "n/d"
    if v < 0:
        return "cold", "Déflation (< 0 %)"
    if v <= 3:
        return "good", "Stable (0–3 %)"
    if v <= thr:
        return "warn", f"Élevée (3–{thr} %)"
    return "bad", f"Très élevée (> {thr} %)"


def r_words(r):
    if pd.isna(r):
        return "n/d"
    if abs(r) < 0.1:
        return "aucun lien"
    return ("lien faible" if abs(r) < 0.3 else "lien modéré" if abs(r) < 0.5 else "lien fort") + \
        (" (Phillips)" if r < 0 else " (inverse)")


def num(v, fmt="{:.1f}", unit=" %"):
    return "n/d" if pd.isna(v) else fmt.format(v).replace(".", ",").replace("-", "−") + unit


def trend(v, bad_if_up=True):
    """Flèche colorée : rouge si la variation va dans le mauvais sens."""
    if pd.isna(v):
        return "n/d"
    if abs(v) < 0.05:
        return "stable"
    color = STATUS["bad" if (v > 0) == bad_if_up else "good"][0]
    return f'<b style="color:{color}">{"▲" if v > 0 else "▼"} {num(v, "{:+.1f}", " pt")}</b>'


# ================================================================ widgets
# Filtres globaux (barre latérale)
region = pn.widgets.MultiChoice(name="Régions", options=sorted(PANEL.region.unique()), placeholder="Toutes")
income = pn.widgets.MultiChoice(name="Groupes de revenu", options=list(INCOME_COLORS), placeholder="Tous")
countries = pn.widgets.MultiChoice(name="Pays", options={n: iso for iso, n in NAMES.items()}, placeholder="Tous",
                                   max_items=8)  # au-delà de 8, plus assez de couleurs distinctes pour les trajectoires
reset = pn.widgets.Button(name="Réinitialiser", button_type="light", button_style="outline")

# Filtres locaux (dans un onglet)
TOGGLE = dict(button_type="primary", button_style="outline", sizing_mode="fixed")
# Onglet Pays : boutons pour passer d'un pays choisi à l'autre (couleur = celle du pays dans « Le lien »)
focus = pn.widgets.RadioButtonGroup(options={}, value=None, visible=False, button_type="primary", button_style="outline",
                                    stylesheets=["".join(f".bk-btn:nth-child({k + 1}) {{ border-bottom:3px solid {c} !important }}"
                                                         for k, c in enumerate(PATH_COLORS))])
map_metric = pn.widgets.RadioButtonGroup(options=["Chômage", "Inflation"], value="Inflation", width=220, **TOGGLE)
# Année affichée : une seule pour tout le tableau (indicateurs, carte, classement, alertes)
year = pn.widgets.IntSlider(name="Année", start=int(PANEL.year.min()), end=int(PANEL.year.max()),
                            value=int(PANEL.year.max()))
# Seuils d'alerte : réglables dans l'onglet Alertes seulement
infl_thr = pn.widgets.IntSlider(name="Alerte si inflation > (%)", start=4, end=30, value=10, width=240,
                                sizing_mode="fixed")
unemp_thr = pn.widgets.FloatSlider(name="et chômage en hausse de plus de (pts)", start=0, end=2, step=0.1, value=0.2,
                                   width=260, sizing_mode="fixed")
only_alerts = pn.widgets.Checkbox(name="Seulement les pays en alerte", value=True, width=240, sizing_mode="fixed",
                                  margin=(12, 12))


def reset_filters(_):
    region.value, income.value, countries.value = [], [], []


reset.on_click(reset_filters)


@pn.depends(countries, watch=True)
def follow_selection(chosen):
    """L'onglet Pays suit le filtre Pays : un bouton par pays choisi, le dernier ajouté est affiché."""
    keep = focus.value if focus.value in chosen else (chosen[-1] if chosen else None)
    focus.param.update(options={NAMES[i]: i for i in chosen}, value=keep, visible=len(chosen) > 1)
    country_charts.visible = bool(chosen)


# ================================================================ calculs
def prep():
    """Toutes les années et tous les pays ; `infl` = inflation globale (alimentaire et énergie : fiche pays)."""
    d = PANEL.copy()
    # hyperinflation : on masque la valeur (NaN) plutôt que de supprimer la ligne, le chômage reste visible
    d[list(MEASURES.values())] = d[list(MEASURES.values())].mask(lambda x: x > HYPER)
    d["infl"] = d["inflation"]
    return d


def apply_groups(d, region, income, countries):
    for col, chosen in [("region", region), ("income_group", income), ("iso3", countries)]:
        if chosen:
            d = d[d[col].isin(chosen)]
    return d


def per_country_r(d):
    """Corrélation de Pearson chômage-inflation pour chaque pays (au moins la moitié des années, min. 5)."""
    d = d.dropna(subset=["unemployment", "infl"])
    min_pts = max(5, d.year.nunique() // 2)
    d = d[d.groupby("iso3").year.transform("size") >= min_pts]
    if d.empty:
        return pd.Series(dtype=float, index=pd.Index([], name="iso3"))
    corr = d.groupby("iso3")[["unemployment", "infl"]].corr()  # matrice 2x2 par pays
    return corr.xs("unemployment", level=1)["infl"].dropna()


ALERTS = ["Stagflation", "Inflation très élevée", "Chômage en hausse", "Aucune"]  # ordre de gravité
ALERT_LEVEL = dict(zip(ALERTS, ["bad", "bad", "warn", "good"]))


LAST_YEAR = int(PANEL.year.max())


def alerts(base, d, infl_thr, unemp_thr, year=LAST_YEAR):
    """Alertes d'une année : niveau, variation du chômage, écart à la région, nombre d'années en stagflation."""
    base = base.sort_values(["iso3", "year"]).assign(delta=lambda x: x.groupby("iso3").unemployment.diff())
    history = ((base.infl > infl_thr) & (base.delta > unemp_thr)).groupby(base.iso3).sum()
    last = base[base.year == year].copy()
    last["n_stag"] = last.iso3.map(history)
    last["vs_region"] = last.infl - last.groupby("region").infl.transform("median")
    hot, rising = last.infl > infl_thr, last.delta > unemp_thr
    last["alert"] = pd.Categorical(np.select([hot & rising, hot, rising], ALERTS[:3], ALERTS[3]),
                                   categories=ALERTS, ordered=True)
    last = last[last.iso3.isin(d.iso3)]  # filtres de groupe appliqués après le calcul des médianes régionales
    return last.sort_values(["alert", "infl"], ascending=[True, False])


def stagflation_by_year(d, infl_thr, unemp_thr):
    d = d.sort_values("year")
    rising = d.groupby("iso3").unemployment.diff() > unemp_thr
    return ((d.infl > infl_thr) & rising).groupby(d.year).sum().iloc[1:]  # 1re année : pas de variation


def peers(iso, n=3):
    if iso not in PROFILE_Z.index:
        return []
    dist = ((PROFILE_Z - PROFILE_Z.loc[iso]) ** 2).sum(axis=1).drop(iso)
    return [NAMES[i] for i in dist.nsmallest(n).index]


BASE = prep()  # tous les pays : médianes régionales, alertes, fiches
data = pn.bind(apply_groups, BASE, region, income, countries)  # sélection filtrée par la barre latérale

# ================================================================ cartes KPI, mini-graphiques et phrases
CSS = f"""
:host {{ color:{TEXT}; font-family:{FONT} }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px }}
.kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px 8px; position:relative }}
.kpi::before {{ content:""; position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; background:var(--c) }}
.kpi .t {{ font-size:11px; color:{TEXT3}; text-transform:uppercase; letter-spacing:.08em; font-weight:600 }}
.kpi .row {{ display:flex; align-items:baseline; gap:10px; margin-top:4px; flex-wrap:wrap }}
.kpi .v {{ font-size:28px; font-weight:700; color:var(--c); letter-spacing:-.02em; line-height:1.2 }}
.kpi .b {{ font-size:11.5px; font-weight:600; padding:2px 8px; border-radius:999px; color:var(--c);
          background:color-mix(in srgb, var(--c) 16%, transparent) }}
.kpi .x {{ font-size:12.5px; color:{TEXT2}; margin-top:4px; min-height:18px }}
.spark {{ display:block; width:100%; height:34px; margin-top:6px }}
.scale {{ display:flex; justify-content:space-between; font-size:10px; color:{TEXT3}; margin-top:-2px }}
.tip {{ background:color-mix(in srgb, {ACCENT} 10%, transparent); border:1px solid color-mix(in srgb, {ACCENT} 30%, transparent);
        padding:9px 14px; border-radius:10px; font-size:13.5px; color:{TEXT2}; line-height:1.5 }}
.tip b {{ color:{TEXT} }}
.hero h1 {{ font-size:22px; font-weight:700; margin:0; letter-spacing:-.01em; color:{TEXT} }}
.hero p {{ margin:4px 0 0; color:{TEXT3}; font-size:13px }}
.sub {{ color:{TEXT3}; font-size:13px; padding-top:10px }}
.lbl {{ font-size:11px; color:{TEXT3}; text-transform:uppercase; letter-spacing:.08em; font-weight:600; margin:6px 0 0 }}
table.p {{ border-collapse:collapse; width:100%; font-size:13px; color:{TEXT2} }}
table.p td, table.p th {{ padding:9px 8px; border-bottom:1px solid {BORDER}; text-align:right }}
table.p th {{ color:{TEXT3}; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.06em }}
table.p td:first-child, table.p th:first-child {{ text-align:left; color:{TEXT} }}
.chip {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px }}
.legend div {{ margin:5px 0; font-size:12.5px; color:{TEXT2} }}
"""


def html(obj, **kw):
    return pn.pane.HTML(obj, stylesheets=[CSS], **kw)


def spark(values, color, mark=None):
    """Mini-courbe SVG sur 1991–2021 ; un trait vertical repère l'année affichée (`mark`)."""
    v = pd.Series(values).dropna()
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    xs = np.linspace(0, 100, len(v))
    ys = 31 - (v.values - lo) / ((hi - lo) or 1) * 28
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    marker = ""
    if mark in v.index:
        mx = xs[list(v.index).index(mark)]
        marker = (f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="0" y2="34" stroke="{TEXT}" stroke-width="1.5" '
                  f'opacity=".8" vector-effect="non-scaling-stroke"/>')
    return (f'<svg class="spark" viewBox="0 0 100 34" preserveAspectRatio="none">'
            f'<polygon points="0,34 {pts} 100,34" fill="{color}" opacity=".12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'vector-effect="non-scaling-stroke"/>{marker}</svg>')


def gauge(r):
    """Jauge de −1 à +1 pour un coefficient de corrélation."""
    if pd.isna(r):
        return ""
    x = (r + 1) * 50
    return (f'<svg class="spark" viewBox="0 0 100 34" preserveAspectRatio="none">'
            f'<rect x="0" y="15" width="50" height="4" fill="{BLUE}" opacity=".35"/>'
            f'<rect x="50" y="15" width="50" height="4" fill="{RED}" opacity=".35"/>'
            f'<rect x="49.7" y="11" width=".6" height="12" fill="{TEXT3}"/>'
            f'<rect x="{x - 1:.1f}" y="7" width="2" height="20" rx="1" fill="{TEXT}"/></svg>'
            f'<div class="scale"><span>−1 Phillips</span><span>0</span><span>+1</span></div>')


def kpi(title, value, level, badge="", context="", chart=""):
    color, icon = STATUS[level]
    badge = f'<span class="b">{icon} {badge.split(" (")[0]}</span>' if badge else ""
    size = "font-size:20px" if len(value) > 10 else ""
    return (f'<div class="kpi" style="--c:{color}"><div class="t">{title}</div>'
            f'<div class="row"><span class="v" style="{size}">{value}</span>{badge}</div>'
            f'<div class="x">{context}</div>{chart}</div>')


def tip(text):
    return f'<div class="tip">{text}</div>' if text else ""


def colored(v, level_fn):
    return f'<b style="color:{STATUS[level_fn(v)[0]][0]}">{num(v)}</b>'


def legend_html():
    rows = [("good", "chômage < 5 % · inflation 0–3 %"), ("warn", f"chômage 5–10 % · inflation 3–{HIGH_INFL} %"),
            ("bad", f"chômage ≥ 10 % · inflation > {HIGH_INFL} %"), ("cold", "déflation (< 0 %)")]
    items = "".join(f'<div><b style="color:{STATUS[k][0]}">{STATUS[k][1]}</b>&nbsp; {t}</div>' for k, t in rows)
    return f'<div class="legend">{items}<div><b style="color:{BLUE}">●</b>&nbsp; r &lt; 0 : courbe de Phillips</div></div>'


def r_window(y):
    """Fenêtre de 10 ans finissant en `y` (r se calcule sur une série d'années, pas sur une seule)."""
    a = max(int(PANEL.year.min()), y - 9)
    return a, a + 9


def kpis_html(d, base, infl_thr, unemp_thr, y):
    if d.empty:
        return tip("Aucun pays ne correspond aux filtres.")
    by_year = d.groupby("year")[["unemployment", "infl"]].median()
    u, i = by_year.unemployment.get(y, np.nan), by_year.infl.get(y, np.nan)
    u_prev, i_prev = by_year.unemployment.get(y - 1, np.nan), by_year.infl.get(y - 1, np.nan)
    vs_prev = (lambda v: f"{trend(v)} vs {y - 1}") if y - 1 in by_year.index else (lambda v: "première année")
    a, b = r_window(y)
    r = per_country_r(d[d.year.between(a, b)])
    stag = alerts(base, d, infl_thr, unemp_thr, y).query("alert == 'Stagflation'")
    lu, li, ls = unemp_level(u), infl_level(i, HIGH_INFL), ("bad" if len(stag) else "good")
    return '<div class="grid">' + "".join([
        kpi(f"Chômage médian · {y}", num(u), *lu, vs_prev(u - u_prev), spark(by_year.unemployment, STATUS[lu[0]][0], y)),
        kpi(f"Inflation · {y}", num(i), *li, f"{vs_prev(i - i_prev)} · cible {TARGET} %",
            spark(by_year.infl, STATUS[li[0]][0], y)),
        kpi(f"Lien chômage–inflation · {a}–{b}", num(r.median(), "{:+.2f}", ""), "info", r_words(r.median()),
            f"r &lt; 0 dans {(r < 0).sum()} pays sur {len(r)}", gauge(r.median())),
        kpi(f"Stagflation · {y}", f"{len(stag)} pays", ls, "À surveiller" if len(stag) else "Aucun",
            ", ".join(stag.country.head(2)) + ("…" if len(stag) > 2 else ""),
            spark(stagflation_by_year(d, infl_thr, unemp_thr), STATUS[ls][0], y)),
    ]) + "</div>"


def world_tip(d, metric, year):
    col, fn = ("unemployment", unemp_level) if metric == "Chômage" else ("infl", lambda v: infl_level(v, HIGH_INFL))
    m = d[d.year == year].dropna(subset=[col])
    if m.empty:
        return ""
    bad = m[col].map(lambda v: fn(v)[0] == "bad").sum()
    return tip(f"En {year}, <b>{bad} pays sur {len(m)}</b> sont en zone ⚠. Cliquez un pays pour ouvrir sa fiche, "
               f"ou ▶ pour animer les années.")


# Hauteurs de l'onglet « Le lien » : les deux cartes de droite (+ marges internes 24 px chacune, écart 14 px)
# ont exactement la hauteur de la carte du nuage, pour que les bas de colonnes s'alignent.
PHILLIPS_H = 620
SIDE_H = (PHILLIPS_H + 24 - 14) // 2 - 24  # = 291 : 2 × (291 + 24) + 14 = 620 + 24

SMALL = 10  # en dessous, une sélection est trop petite pour des statistiques sur la distribution des pays


def decade_story(d):
    """Décrit les trois décennies (pas seulement la première et la dernière)."""
    dec = [(f"{a}–{b}", per_country_r(d[d.year.between(a, b)]).median()) for a, b in PERIODS]
    dec = [(p, v) for p, v in dec if pd.notna(v)]
    if len(dec) < 2:
        return ""
    values = " → ".join(f"{num(v, '{:+.2f}', '')} ({p})" for p, v in dec)
    i = min(range(len(dec)), key=lambda k: dec[k][1])  # décennie où le lien de Phillips est le plus marqué
    (p_best, best), (_, last) = dec[i], dec[-1]
    if best >= -0.1:
        verdict = "aucune décennie ne montre de lien net"
    elif i < len(dec) - 1 and abs(last) < abs(best) / 2:
        verdict = f"le lien est le plus marqué en {p_best}, puis s'affaiblit"
    else:
        verdict = f"le lien est le plus marqué en {p_best}"
    return f" Par décennie : {values} ; {verdict}."


def link_tip(d):
    dd = d.dropna(subset=["unemployment", "infl"])
    r_all = dd.unemployment.corr(dd.infl) if len(dd) > 2 else np.nan
    r = per_country_r(d)
    strong = (r <= -0.5).sum()
    warn = (f"<b style='color:{STATUS['warn'][0]}'>⚠ Seulement {len(r)} pays sélectionné{'s' if len(r) > 1 else ''} :"
            f" à lire avec prudence.</b> " if len(r) < SMALL else "")
    return tip(f"{warn}Tous pays confondus : <b>{r_words(r_all)}</b> (r = {num(r_all, '{:+.2f}', '')}). Pays par pays, "
               f"<b>{(r < 0).mean() * 100 if len(r) else 0:.0f} %</b> suivent la courbe de Phillips, "
               f"{'dont ' + str(strong) + ' fortement' if strong else 'aucun fortement'}.{decade_story(d)}")


def profiles_html(d):
    p = PROFILES[PROFILES.iso3.isin(d.iso3)]
    g = p.groupby("profile").agg(n=("iso3", "size"), u=("unemp_median", "median"), i=("infl_median", "median"),
                                 r=("phillips_r", "median")).reindex(list(PROFILE_COLORS)).dropna(subset=["n"])
    rows = "".join(
        f'<tr><td><span class="chip" style="background:{PROFILE_COLORS[k]}"></span>{k}</td><td>{int(x.n)}</td>'
        f'<td>{colored(x.u, unemp_level)}</td><td>{colored(x.i, lambda v: infl_level(v, HIGH_INFL))}</td>'
        f'<td>{num(x.r, "{:+.2f}", "")}</td></tr>' for k, x in g.iterrows())
    return ('<table class="p"><tr><th>Profil · médianes 1991–2021</th><th>Pays</th><th>Chômage</th>'
            f'<th>Inflation</th><th>r</th></tr>{rows}</table>')


def peers_tip(chosen):
    if not chosen:
        return tip("Ajoutez des pays au filtre <b>Pays</b> pour voir leurs plus proches voisins de profil.")
    lines = [f"<b>{NAMES[i]}</b> → {', '.join(peers(i))}" for i in chosen if peers(i)]
    return tip("Pays les plus semblables : " + " · ".join(lines) + ".") if lines else ""


def country_empty(chosen):
    return "" if chosen else tip("Aucun pays choisi. Ajoutez-en dans le filtre <b>Pays</b> de la barre latérale, ou cliquez "
                                 "un pays sur la carte, dans le classement, un nuage ou le tableau d'alertes.")


def country_sub(iso):
    if iso not in NAMES.index:
        return ""
    info = PANEL[PANEL.iso3 == iso].iloc[0]
    return f'<div class="sub">{info.region} · {info.income_group}</div>'


def category_tip(base, iso):
    c = base[base.iso3 == iso].dropna(subset=["inflation"])
    if c.empty:
        return ""
    row = c.iloc[-1]
    parts = {k: row[v] for k, v in MEASURES.items() if pd.notna(row[v])}
    others = {k: v for k, v in parts.items() if k != "Globale"}
    if not others:
        return tip("Seule l'inflation globale est disponible pour ce pays.")
    k, v = max(others.items(), key=lambda x: x[1])
    if v - parts["Globale"] > 1:
        return tip(f"En {int(row.year)}, c'est <b>{'l’alimentation' if k == 'Alimentaire' else 'l’énergie'}</b> "
                   f"({num(v, '{:+.1f}')}) qui tire l'inflation ({num(parts['Globale'], '{:+.1f}')}).")
    return tip(f"En {int(row.year)}, ni l'alimentation ni l'énergie ne tirent l'inflation "
               f"({num(parts['Globale'], '{:+.1f}')}).")


def alerts_tip(base, d, infl_thr, unemp_thr, year):
    n = alerts(base, d, infl_thr, unemp_thr, year).alert.value_counts()
    per_year = stagflation_by_year(d, infl_thr, unemp_thr)
    peak = f" Pic : <b>{per_year.idxmax()}</b> ({per_year.max()} pays)." if len(per_year) and per_year.max() else ""
    return tip(f"{year} : <b style='color:{STATUS['bad'][0]}'>{n.get('Stagflation', 0)} pays en stagflation</b>"
               f" (inflation > {infl_thr} % et chômage en hausse) · {n.get('Inflation très élevée', 0)} en inflation "
               f"très élevée · {n.get('Chômage en hausse', 0)} avec chômage en hausse.{peak}")


# ================================================================ figures
def style(fig, title, height=420):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family=FONT, size=12, color=TEXT2), margin=dict(l=12, r=12, t=44, b=12),
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01, y=0.97),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title_text="", font_color=TEXT2),
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def empty(msg="Aucune donnée pour cette sélection"):
    return style(px.scatter().add_annotation(text=msg, showarrow=False, font=dict(size=14, color=TEXT3))
                 .update_xaxes(visible=False).update_yaxes(visible=False), "")


def band(fig, y0, y1, **kw):
    """Zone « saine » en vert discret."""
    fig.add_hrect(y0=y0, y1=y1, fillcolor=STATUS["good"][0], opacity=0.09, line_width=0, **kw)


def level_frame(m, col):
    """Ajoute le libellé de niveau (Faible/Modéré/…) ; renvoie aussi l'ordre et les couleurs des niveaux."""
    fn = unemp_level if col == "unemployment" else (lambda v: infl_level(v, HIGH_INFL))
    reps = [1, 7, 12] if col == "unemployment" else [-1, 1, 4, HIGH_INFL + 1]
    order = [fn(v)[1] for v in reps]
    colors = {fn(v)[1]: STATUS[fn(v)[0]][0] for v in reps}
    return m.assign(level=m[col].map(lambda v: fn(v)[1])), order, colors


def map_fig(d, metric, year):
    if d.empty:
        return empty()
    hover = dict(hover_name="country", locations="iso3")
    if metric == "Corrélation r":
        m = per_country_r(d).rename("r").reset_index().merge(PROFILES[["iso3", "country"]])
        fig = px.choropleth(m, color="r", color_continuous_scale=DIVERGING, range_color=(-1, 1), **hover)
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.6, outlinewidth=0))
        title = f"r par pays · {d.year.min()}–{d.year.max()} · bleu = Phillips, rouge = inverse"
    else:
        col = "unemployment" if metric == "Chômage" else "infl"
        m, order, colors = level_frame(d[d.year == year].dropna(subset=[col]), col)
        fig = px.choropleth(m, color="level", color_discrete_map=colors, category_orders={"level": order},
                            hover_data={col: ":.1f", "level": False, "iso3": False},
                            labels={col: f"{metric} (%)"}, **hover)
        title = f"{metric} · {year}"
    fig.update_traces(marker_line_color=BG, marker_line_width=0.4, marker_opacity=0.85)  # moins criard sur fond sombre
    fig.update_geos(showframe=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", showland=True,
                    landcolor=SURFACE2, showcountries=False, showcoastlines=False, showlakes=False)
    return style(fig, title, 480).update_layout(legend=dict(y=0.02, yanchor="bottom"))


def ranking_fig(d, metric, year):
    if d.empty:
        return empty()
    if metric in ("Chômage", "Inflation"):
        col = "unemployment" if metric == "Chômage" else "infl"
        m, order, colors = level_frame(d[d.year == year].dropna(subset=[col]).nlargest(10, col), col)
        if m.empty:
            return empty()
        fig = px.bar(m, x=col, y="country", orientation="h", color="level", color_discrete_map=colors,
                     category_orders={"level": order}, custom_data=["iso3"], text=m[col].map(num),
                     labels={col: "%", "country": ""})
        title, ys = f"Top 10 · {year}", list(m.country)
    else:
        r = per_country_r(d).nsmallest(10).rename("r").reset_index()
        if r.empty:
            return empty()
        r["country"] = r.iso3.map(NAMES)
        fig = px.bar(r, x="r", y="country", orientation="h", custom_data=["iso3"], color_discrete_sequence=[BLUE],
                     text=r.r.map(lambda v: num(v, "{:+.2f}", "")), labels={"country": ""})
        title, ys = "Liens de Phillips les plus forts", list(r.country)
    fig.update_yaxes(categoryorder="array", categoryarray=ys, autorange="reversed", showgrid=False)
    xs = np.concatenate([t.x for t in fig.data])
    fig.update_xaxes(range=[min(min(xs), 0) * 1.3, max(max(xs), 0) * 1.25], showgrid=False, visible=False)
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=TEXT2, marker_cornerradius=4)
    if metric == "Corrélation r":  # barres négatives : l'étiquette dehors chevaucherait le nom du pays
        fig.update_traces(textposition="inside", insidetextanchor="start", textfont_color=TEXT)
    return style(fig, title, 480).update_layout(showlegend=False, bargap=0.35)


def trend_line(fig, x, y, name, legend=True, x_span=None, **line):
    """Droite des moindres carrés (tracée sur x_span, sinon l'étendue des x) ; la légende porte le r."""
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array(x_span if x_span is not None else [x.min(), x.max()])
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", hoverinfo="skip", line=line, showlegend=legend,
                    name=f"{name} (r = {num(x.corr(y), '{:+.2f}', '')})")


def hidden_note(iso):
    raw = PANEL.loc[PANEL.iso3 == iso, "inflation"]
    n = int((raw > HYPER).sum())
    return f" · {n} an{'s' if n > 1 else ''} masqué{'s' if n > 1 else ''} (> {HYPER} %)" if n else ""


DECADE_COLORS = [BLUE, ORANGE, AQUA]  # 1991–2000, 2001–2010, 2011–2021


def add_dots(fig, c, iso, color, name):
    """Un point par année, sans trait (30 années reliées deviennent illisibles) ; l'année s'affiche au survol."""
    fig.add_scatter(x=c.unemployment, y=c.infl, mode="markers", name=name, customdata=[[iso, y] for y in c.year],
                    marker=dict(size=10, color=color, opacity=0.9, line=dict(color=BG, width=1)),
                    hovertemplate=f"<b>{NAMES[iso]}</b> %{{customdata[1]}}<br>chômage %{{x:.1f}} %"
                                  "<br>inflation %{y:.1f} %<extra></extra>")


def corner_badge(fig, text):
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="top", showarrow=False,
                       align="right", font=dict(size=13, color=TEXT), bgcolor=SURFACE2, bordercolor=BORDER,
                       borderpad=8, text=text)


def subtitle(text):
    return f"Chômage vs inflation<br><span style='font-size:12px;color:{TEXT3}'>{text}</span>"


def phillips_fig(d, base, chosen):
    d = d.dropna(subset=["unemployment", "infl"])
    if len(d) < 3:
        return empty()
    if chosen:
        fig = px.scatter()
        if len(chosen) == 1:  # un pays : une couleur par décennie, pour voir si le lien change avec le temps
            iso = chosen[0]
            c = d[d.iso3 == iso]
            for (a, b), color in zip(PERIODS, DECADE_COLORS):
                part = c[c.year.between(a, b)]
                if part.empty:
                    continue
                ok = len(part) >= 5
                add_dots(fig, part, iso, color,
                         f"{a}–{b}" + (f" (r = {num(part.unemployment.corr(part.infl), '{:+.2f}', '')})" if ok else ""))
                if ok:
                    trend_line(fig, part.unemployment, part.infl, "", legend=False, color=color, width=2.5)
            r = c.unemployment.corr(c.infl) if len(c) >= 5 else np.nan
            note = hidden_note(iso).replace(" · ", "")
            corner_badge(fig, f"<b>{NAMES[iso]}</b> · r = {num(r, '{:+.2f}', '')}<br>{r_words(r)}"
                              + (f"<br><span style='font-size:11px;color:{TEXT3}'>{note}</span>" if note else ""))
            sub = "Un point par année · couleur = décennie · pente ↘ = courbe de Phillips"
        else:  # plusieurs pays : une couleur par pays, une droite de tendance chacun
            for k, iso in enumerate(chosen):
                c = d[d.iso3 == iso]
                if c.empty:
                    continue
                color, ok = PATH_COLORS[k % len(PATH_COLORS)], len(c) >= 5
                add_dots(fig, c, iso, color,
                         NAMES[iso] + (f" (r = {num(c.unemployment.corr(c.infl), '{:+.2f}', '')})" if ok else ""))
                if ok:
                    trend_line(fig, c.unemployment, c.infl, "", legend=False, color=color, width=2.5)
            sub = "Un point par année · couleur = pays · pente ↘ = courbe de Phillips"
        fig.update_xaxes(title_text="Chômage (%)")
        fig.update_yaxes(title_text="Inflation (%)")
        return style(fig, subtitle(sub), PHILLIPS_H).update_layout(uirevision="|".join(chosen), margin_t=70)
    pts = d.groupby(["iso3", "country"], as_index=False)[["unemployment", "infl"]].mean()  # un point par pays
    fig = px.scatter(pts, x="unemployment", y="infl", hover_name="country", custom_data=["iso3"], opacity=0.8,
                     color_discrete_sequence=[GRAY], labels={"unemployment": "Chômage (%)", "infl": "Inflation (%)"})
    fig.update_traces(marker=dict(size=8, line_width=0))
    world = base.dropna(subset=["unemployment", "infl"])
    trend_line(fig, world.unemployment, world.infl, "Tendance mondiale", x_span=[pts.unemployment.min(),
               pts.unemployment.max()], color=TEXT3, dash="dash", width=1.5)
    lo, hi = pts.infl.quantile([0.01, 0.99])  # on coupe les 1 % extrêmes pour que le nuage reste lisible
    fig.update_yaxes(range=[lo - 2, hi + 2])
    sub = "Un point par pays (moyenne 1991–2021) · pente ↘ = courbe de Phillips · cliquez un point pour le détailler"
    # uirevision change avec la vue : Plotly réinitialise alors zoom et axes au lieu de garder ceux de la vue précédente
    return style(fig, subtitle(sub), PHILLIPS_H).update_layout(uirevision="tous", margin_t=70)


def r_hist_fig(d, chosen):
    r = per_country_r(d)
    if r.empty:
        return empty()
    if len(r) < SMALL:  # trop peu de pays pour un histogramme : une barre par pays, triée
        df = r.sort_values().rename("r").reset_index()
        df["country"] = df.iso3.map(NAMES)
        # même couleur par pays que les trajectoires du nuage quand des pays sont choisis
        colors = [PATH_COLORS[list(chosen).index(i) % len(PATH_COLORS)] if i in chosen
                  else (BLUE if v < 0 else GRAY) for i, v in zip(df.iso3, df.r)]
        fig = px.bar(df, x="r", y="country", orientation="h", text=df.r.map(lambda v: num(v, "{:+.2f}", "")),
                     labels={"r": "r", "country": ""})
        fig.update_traces(marker_color=colors, marker_line_width=0, textposition="outside", cliponaxis=False,
                          textfont_color=TEXT2, marker_cornerradius=3)
        fig.update_xaxes(range=[-1, 1], zeroline=True)
        fig.update_yaxes(categoryorder="array", categoryarray=list(df.country), autorange="reversed")
        return style(fig, f"r par pays · {len(r)} pays", SIDE_H).update_layout(bargap=0.35)
    df = r.rename("r").reset_index()
    df["sign"] = np.where(df.r < 0, "r < 0", "r ≥ 0")
    fig = px.histogram(df, x="r", color="sign", nbins=20, range_x=(-1, 1),
                       color_discrete_map={"r < 0": BLUE, "r ≥ 0": GRAY}, labels={"r": "r", "sign": ""})
    fig.add_vline(x=r.median(), line=dict(color=TEXT3, dash="dot"))
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis_title="Pays", bargap=0.12, showlegend=False)
    return style(fig, f"Un r par pays · {len(r)} pays", SIDE_H)


def decade_fig(d):
    rows = [(f"{a}–{b}", per_country_r(d[d.year.between(a, b)]).median()) for a, b in PERIODS]
    df = pd.DataFrame(rows, columns=["period", "r"]).dropna()
    if df.empty:
        return empty("Période trop courte")
    fig = px.bar(df, x="period", y="r", text=df.r.map(lambda v: num(v, "{:+.2f}", "")),
                 labels={"period": "", "r": "r médian"}, color_discrete_sequence=[BLUE])
    fig.update_traces(textposition="outside", width=0.45, cliponaxis=False, textfont_color=TEXT2,
                      marker_cornerradius=4)
    pad = max(0.05, df.r.abs().max() * 0.4)  # marge pour que les petites barres restent visibles
    fig.update_yaxes(range=[min(df.r.min(), 0) - pad, max(df.r.max(), 0) + pad])
    return style(fig, "r médian par décennie", SIDE_H)


def profile_fig(d):
    p = PROFILES[PROFILES.iso3.isin(d.iso3) & (PROFILES.infl_median > 0)]
    if p.empty:
        return empty()
    fig = px.scatter(p, x="unemp_median", y="infl_median", color="profile", color_discrete_map=PROFILE_COLORS,
                     category_orders={"profile": list(PROFILE_COLORS)}, hover_name="country", custom_data=["iso3"],
                     log_y=True, hover_data={"phillips_r": ":+.2f"},
                     labels={"unemp_median": "Chômage médian (%)", "infl_median": "Inflation médiane (%, log)",
                             "profile": "Profil", "phillips_r": "r"})
    fig.update_traces(marker=dict(size=10, line=dict(color=BG, width=1)))
    fig.add_vrect(x0=0, x1=5, fillcolor=STATUS["good"][0], opacity=0.07, line_width=0)  # zones saines
    band(fig, 0.5, 3)  # (unités de données, même en échelle log)
    return style(fig, "Un point par pays", 520).update_layout(showlegend=False)


def country_fig(base, iso):
    c = base[base.iso3 == iso]
    if c.empty:
        return empty("")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    band(fig, 0, 5, row=1, col=1)
    band(fig, 0, 3, row=2, col=1)
    fig.add_hline(y=HIGH_INFL, line=dict(color=STATUS["bad"][0], dash="dash", width=1), row=2, col=1)
    fig.add_scatter(x=c.year, y=c.unemployment, name=NAMES[iso], line=dict(color=TEXT, width=2.5), row=1, col=1)
    fig.add_scatter(x=c.year, y=c.infl, line=dict(color=TEXT, width=2.5), showlegend=False, row=2, col=1)
    reg = base[base.region == c.region.iloc[0]].groupby("year")[["unemployment", "infl"]].median()
    line = dict(color=BLUE, width=2, dash="dot")
    fig.add_scatter(x=reg.index, y=reg.unemployment, name="Médiane de la région", line=line, row=1, col=1)
    fig.add_scatter(x=reg.index, y=reg.infl, line=line, showlegend=False, row=2, col=1)
    for y, label in SHOCKS.items():
        if c.year.min() <= y <= c.year.max():
            fig.add_vline(x=y, line=dict(color=BORDER, width=1.5))
            fig.add_annotation(x=y, y=1.0, yref="paper", text=label, showarrow=False, yshift=12,
                               font=dict(size=10, color=TEXT3))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")  # la bande verte (zone saine) reste visible
    fig.update_yaxes(title_text="Chômage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Inflation (%)", row=2, col=1)
    return style(fig, "", 520).update_layout(margin_t=36, legend=dict(y=-0.07))


def category_fig(base, iso):
    c = base[base.iso3 == iso]
    if c.empty:
        return empty("")
    c = c[c.year > c.year.max() - 10]  # 10 dernières années : lisible
    long = c.melt(id_vars="year", value_vars=list(MEASURES.values()), var_name="serie", value_name="val").dropna()
    if long.empty:
        return empty("Pas de données d'inflation")
    long["serie"] = long.serie.map({v: k for k, v in MEASURES.items()})
    fig = px.bar(long, x="year", y="val", color="serie", barmode="group", color_discrete_map=SERIES_COLORS,
                 category_orders={"serie": list(MEASURES)}, labels={"year": "", "val": "Inflation (%)"})
    band(fig, 0, 3)
    fig.add_hline(y=HIGH_INFL, line=dict(color=STATUS["bad"][0], dash="dash", width=1))
    fig.update_traces(marker_line_width=0, marker_cornerradius=2)
    fig.update_layout(bargap=0.25, bargroupgap=0.08)
    return style(fig, "Inflation par catégorie", 470).update_layout(legend=dict(y=-0.08))


def stag_fig(d, infl_thr, unemp_thr, year):
    """Nombre de pays en stagflation chaque année ; l'année affichée dans le tableau est mise en évidence."""
    s = stagflation_by_year(d, infl_thr, unemp_thr)
    if s.empty:
        return empty()
    fig = px.bar(x=s.index, y=s.values, labels={"x": "", "y": "Pays en stagflation"})
    fig.update_traces(marker_color=[STATUS["bad"][0] if y == year else "#5b2e30" for y in s.index],
                      marker_line_width=0, marker_cornerradius=3,
                      hovertemplate="%{x} : %{y} pays<extra></extra>")
    for y, label in SHOCKS.items():
        fig.add_annotation(x=y, y=s.max(), text=label, showarrow=False, yshift=16, font=dict(size=10, color=TEXT3))
    return style(fig, f"Pays en stagflation chaque année · cliquez une barre pour voir l'année", 260).update_layout(
        bargap=0.25, margin_t=60)


# ================================================================ tableau d'alertes
def alert_view(base, d, infl_thr, unemp_thr, only, year):
    a = alerts(base, d, infl_thr, unemp_thr, year)
    if only:
        a = a[a.alert != "Aucune"]
    a = a.assign(alert=a.alert.astype(str).map(lambda s: f"{STATUS[ALERT_LEVEL[s]][1]} {s}"))
    return a[["iso3", "alert", "country", "region", "infl", "unemployment", "delta", "vs_region", "n_stag"]].rename(columns={
        "alert": "Alerte", "country": "Pays", "region": "Région", "infl": "Inflation (%)",
        "unemployment": "Chômage (%)", "delta": "Δ chômage 1 an", "vs_region": "Écart région",
        "n_stag": "Années en stagflation"}
    ).round(2).reset_index(drop=True)


def cell_colors(df):
    """Couleur de texte selon le niveau de chaque valeur (mêmes repères que le reste du tableau de bord)."""
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    text = lambda lvl: f"color: {STATUS[lvl][0]}; font-weight: 600"
    out["Inflation (%)"] = df["Inflation (%)"].map(lambda v: text(infl_level(v, HIGH_INFL)[0]))
    out["Chômage (%)"] = df["Chômage (%)"].map(lambda v: text(unemp_level(v)[0]))
    out["Δ chômage 1 an"] = df["Δ chômage 1 an"].map(
        lambda v: text("bad" if v > unemp_thr.value else "good" if v < 0 else "neutral") if pd.notna(v) else "")
    out["Alerte"] = df["Alerte"].map(lambda s: text(ALERT_LEVEL[s.split(" ", 1)[1]]))
    return out


TABLE_CSS = f"""
.tabulator {{ background:{SURFACE}; border:none; font-family:{FONT}; font-size:13px }}
.tabulator .tabulator-header, .tabulator .tabulator-header .tabulator-col {{ background:{SURFACE}; color:{TEXT3};
  border-color:{BORDER}; font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.05em }}
.tabulator .tabulator-header {{ border-bottom:1px solid {BORDER} }}
.tabulator-row, .tabulator-row.tabulator-row-even {{ background:{SURFACE}; color:{TEXT2}; border-bottom:1px solid {GRID} }}
.tabulator-row:hover {{ background:{SURFACE2} !important; cursor:pointer }}
.tabulator-row .tabulator-cell {{ border-right:none; padding:9px 10px }}
.tabulator .tabulator-footer {{ background:{SURFACE}; border-top:1px solid {BORDER}; color:{TEXT3} }}
.tabulator .tabulator-header-filter input {{ background:{SURFACE2}; color:{TEXT}; border:1px solid {BORDER}; border-radius:6px }}
"""
table = pn.widgets.Tabulator(pn.bind(alert_view, BASE, data, infl_thr, unemp_thr, only_alerts, year),
                             hidden_columns=["iso3"], show_index=False, disabled=True, pagination="local",
                             page_size=15, layout="fit_data_stretch", header_filters={"Pays": {"type": "input"}},
                             widths={"Alerte": 200}, theme="simple", stylesheets=[TABLE_CSS])
table.style.apply(cell_colors, axis=None)
download = pn.widgets.FileDownload(callback=lambda: io.StringIO(table.value.drop(columns="iso3").to_csv(index=False)),
                                   filename="alertes_inflation_chomage.csv", label="Exporter (CSV)",
                                   button_type="primary", width=160, sizing_mode="fixed")

# ================================================================ page
TAB_COUNTRY = 3  # index de l'onglet « Pays »


def select_country(iso, open_tab=False):
    """Un clic sur un pays l'ajoute au filtre Pays (un seul concept de « pays choisis » dans tout le tableau)."""
    if iso not in NAMES.index:
        return
    if iso not in countries.value:
        if len(countries.value) >= countries.max_items:
            return
        countries.value = countries.value + [iso]
    focus.value = iso
    if open_tab:  # exploration -> détail : on ouvre la fiche du pays
        tabs.active = TAB_COUNTRY


def plot(fig_fn, *args, open_tab=None, height=None):
    """Graphique Plotly réactif. open_tab=False : un clic ajoute le pays au filtre ; True : ouvre aussi sa fiche."""
    # height = celle de la figure : sinon Panel réserve une hauteur par défaut et laisse un vide sous le graphique
    pane = pn.pane.Plotly(pn.bind(fig_fn, *args), height=height,
                          config={"displaylogo": False, "displayModeBar": "hover"})
    if open_tab is not None:
        pane.param.watch(lambda e: [select_country(p.get("location") or (p.get("customdata") or [None])[0], open_tab)
                                    for p in ((e.new or {}).get("points") or [])[:1]], "click_data")
    return pane


def box(*objects, flex="1 1 400px"):
    """Carte : fond légèrement plus clair que la page, coins arrondis."""
    return pn.Column(*objects, min_width=260, margin=7, styles={
        "flex": flex, "background": SURFACE, "border": f"1px solid {BORDER}", "border-radius": "12px",
        "padding": "6px 4px"})


def row(*objects):  # passe à la ligne si l'écran est étroit
    return pn.FlexBox(*objects, flex_wrap="wrap", align_items="flex-start")


table.on_click(lambda e: select_country(table.value.iloc[e.row]["iso3"], open_tab=True))

world = pn.Column(
    row(map_metric),
    row(box(plot(map_fig, data, map_metric, year, open_tab=True), flex="2 1 560px"),
        box(plot(ranking_fig, data, map_metric, year, open_tab=True), flex="1 1 300px")),
    html(pn.bind(world_tip, data, map_metric, year), margin=(4, 7)),
)
link = pn.Column(
    html(pn.bind(link_tip, data), margin=(4, 7)),
    row(box(plot(phillips_fig, data, BASE, countries, open_tab=False, height=PHILLIPS_H), flex="3 1 520px"),
        # flex « 0 0 auto » : empilées, les cartes ne doivent pas prendre la base de 400 px (prévue pour la largeur)
        pn.Column(box(plot(r_hist_fig, data, countries, height=SIDE_H), flex="0 0 auto"),
                  box(plot(decade_fig, data, height=SIDE_H), flex="0 0 auto"),
                  styles={"flex": "2 1 300px"}, min_width=260)),
    # carte du r (toute la période, comme le reste de cet onglet) : un clic ajoute le pays au filtre Pays
    row(box(plot(map_fig, data, "Corrélation r", LAST_YEAR, open_tab=False), flex="2 1 560px"),
        box(plot(ranking_fig, data, "Corrélation r", LAST_YEAR, open_tab=False), flex="1 1 300px")),
)
profiles = pn.Column(
    html(pn.bind(peers_tip, countries), margin=(4, 7)),
    row(box(plot(profile_fig, data), flex="3 1 520px"),
        box(html(pn.bind(profiles_html, data), margin=(10, 14)), flex="2 1 340px")),
)
country_charts = row(box(plot(country_fig, BASE, focus), flex="3 1 520px"),
                     box(plot(category_fig, BASE, focus), html(pn.bind(category_tip, BASE, focus), margin=(0, 10, 10, 10)),
                         flex="2 1 340px"))
country_charts.visible = False
country_tab = pn.Column(
    pn.Row(focus, html(pn.bind(country_sub, focus))),
    html(pn.bind(country_empty, countries), margin=(4, 7)),
    country_charts,
)
stag_pane = pn.pane.Plotly(pn.bind(stag_fig, data, infl_thr, unemp_thr, year),
                           config={"displaylogo": False, "displayModeBar": False})
stag_pane.param.watch(lambda e: [setattr(year, "value", int(p["x"]))
                                 for p in ((e.new or {}).get("points") or [])[:1]], "click_data")
alerts_tab = pn.Column(
    html(pn.bind(alerts_tip, BASE, data, infl_thr, unemp_thr, year), margin=(4, 7)),
    pn.FlexBox(infl_thr, unemp_thr, align_items="center"),
    box(stag_pane, flex="1 1 100%"),
    pn.FlexBox(only_alerts, pn.Spacer(), download, align_items="center"),
    box(table, flex="1 1 100%"),
)

TABS_CSS = f"""
.bk-header {{ border-bottom:1px solid {BORDER}; margin-bottom:10px }}
.bk-tab {{ color:{TEXT3}; background:transparent !important; border:none !important; font-family:{FONT};
           font-size:14px; font-weight:500; padding:10px 16px; border-bottom:2px solid transparent !important }}
.bk-tab:hover {{ color:{TEXT} }}
.bk-tab.bk-active {{ color:{TEXT}; border-bottom:2px solid {ACCENT} !important }}
"""
tabs = pn.Tabs(("Monde", world), ("Le lien", link), ("Profils", profiles), ("Pays", country_tab),
               ("Alertes", alerts_tab), dynamic=True, margin=(18, 0, 0, 0), stylesheets=[TABS_CSS])

GUIDE = f"""
## Guide

**Question :** quand le chômage monte, l'inflation baisse-t-elle (courbe de Phillips) ? Partout, et toujours ?

**Onglets** — *Monde* : situation par pays et classement · *Le lien* : la relation chômage–inflation, tous pays,
pays par pays, par décennie · *Profils* : pays semblables (KMeans) · *Pays* : fiche détaillée · *Alertes* : pays à surveiller.

**Interactions** — les filtres de la barre latérale (régions, revenu, pays) s'appliquent à **tous** les onglets.
Cliquer un pays (carte, classement, nuage, tableau) l'ajoute au filtre Pays : *Le lien* montre alors sa trajectoire et
l'onglet *Pays* sa fiche. Retirez-le avec ✕ au-dessus des onglets. Les seuils d'alerte se règlent dans *Alertes*.
L'**année affichée** (barre latérale) règle les indicateurs, la carte, le classement et les alertes ; le lien chômage–inflation y est calculé sur les 10 ans qui finissent cette année-là.
Survol = valeurs, glisser = zoom, double-clic = retour. Période analysée : 1991–2021 ; inflation globale (alimentaire et
énergie dans la fiche pays).

**Repères** — chômage : ✓ < 5 % · ! 5–10 % · ⚠ ≥ 10 %. Inflation : ✓ 0–3 % (cible des banques centrales : {TARGET} %)
· ! 3–{HIGH_INFL} % · ⚠ au-delà de {HIGH_INFL} % · ↓ déflation. ▲▼ rouge = évolution dans le mauvais sens.
r (bleu) entre −1 et 1 : r < 0 = chômage ↑, inflation ↓. |r| < 0,1 aucun lien ; < 0,3 faible ; < 0,5 modéré ; sinon fort.
Corrélation ≠ causalité. *Stagflation* : inflation au-delà du seuil **et** chômage en hausse.

**Données** — chômage : OIT, modélisé (Banque mondiale) ; inflation : Banque mondiale (Ha-Kose-Ohnsorge) ;
régions : API Banque mondiale. 180 pays ; trous ≤ 2 ans interpolés ; inflation > {HYPER} % (hyperinflation) masquée.
"""

guide_btn = pn.widgets.Button(name="Guide", icon="help-circle", button_type="primary", button_style="outline",
                              width=100, sizing_mode="fixed")

CHIP_CSS = f"""
.bk-btn {{ border-radius:999px !important; font:500 12.5px {FONT} !important; padding:0 12px !important;
          background:color-mix(in srgb, {ACCENT} 14%, transparent) !important; color:{TEXT} !important;
          border:1px solid color-mix(in srgb, {ACCENT} 45%, transparent) !important }}
.bk-btn:hover {{ background:color-mix(in srgb, {ACCENT} 28%, transparent) !important }}
"""


def chip(text, on_remove):
    """Pastille d'un filtre actif ; un clic retire ce filtre."""
    b = pn.widgets.Button(name=f"{text}  ✕", sizing_mode="fixed", width=int(len(text) * 7.4) + 46, height=30,
                          margin=(4, 4), stylesheets=[CHIP_CSS], description="Retirer ce filtre")
    b.on_click(lambda _: on_remove())
    return b


def without(widget, item):
    widget.value = [x for x in widget.value if x != item]


def filters_bar(regions, incomes, chosen, d):
    """Toujours visible au-dessus des onglets : ce que montrent TOUS les onglets, et les filtres à retirer."""
    chips = []
    for widget, values in [(region, regions), (income, incomes)]:
        chips += [chip(v, lambda w=widget, v=v: without(w, v)) for v in values]
    chips += [chip(NAMES[iso], lambda iso=iso: without(countries, iso)) for iso in chosen]
    n = d.iso3.nunique()
    if not chips:
        return pn.Spacer(height=0)
    head = html(f'<div class="sub" style="padding-top:9px">Tous les onglets filtrés · <b style="color:{TEXT}">{n} pays'
                '</b></div>', margin=(4, 4), sizing_mode="fixed", width=230)
    clear = pn.widgets.Button(name="Tout effacer", button_type="light", button_style="outline", sizing_mode="fixed",
                              width=110, height=30, margin=(4, 4))
    clear.on_click(reset_filters)
    return pn.FlexBox(head, *chips, clear, align_items="center")


main = pn.Column(
    html('<div class="hero"><h1>Quand le chômage monte, l\'inflation baisse-t-elle ?</h1></div>', margin=(0, 7, 10, 7)),
    pn.panel(pn.bind(filters_bar, region, income, countries, data), margin=(0, 3, 8, 3)),
    html(pn.bind(kpis_html, data, BASE, infl_thr, unemp_thr, year), margin=(0, 7)),
    tabs,
)


def label(text):
    return html(f'<div class="lbl">{text}</div>', margin=(14, 10, 0, 10))


sidebar = [
    label("Année affichée"), year,
    label("Filtres · tous les onglets"), region, income, countries,
    html(f'<div class="sub" style="padding-top:0">Astuce : cliquer un pays sur la carte, dans un nuage ou dans un '
         f'tableau l\'ajoute au filtre Pays.</div>', margin=(0, 10)),
    label("Légende"), html(legend_html(), margin=(4, 10)),
    pn.Spacer(height=10), reset,
]

template = pn.template.FastListTemplate(
    title="Inflation & chômage", sidebar=sidebar, main=[main], header=[guide_btn], theme="dark", theme_toggle=False,
    sidebar_width=300, main_max_width="1500px", main_layout=None, accent_base_color=ACCENT,
    header_background=BG, header_color=TEXT, background_color=BG, neutral_color=SURFACE, corner_radius=10,
    shadow=False, font=FONT, font_url="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
)
template.config.raw_css.append(f"""
#header {{ border-bottom:1px solid {BORDER}; box-shadow:none }}
#sidebar {{ background:{BG}; border-right:1px solid {BORDER} }}
#main {{ background:{BG} }}
""")
template.modal.append(pn.pane.Markdown(GUIDE))
guide_btn.on_click(lambda _: template.open_modal())
template.servable()
