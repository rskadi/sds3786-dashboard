# Inflation et chômage dans le monde (1991–2021)

Tableau de bord interactif (Panel + Plotly) — SDS3786, Automne 2026.
Rayane Slimane-Kadi · Mohamed Elsobky

**Question centrale :** existe-t-il une relation entre l'inflation et le chômage (courbe de Phillips),
et cette relation diffère-t-elle selon les pays et dans le temps ?

## Lancer

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_data.py      # data/raw -> data/clean
.venv/bin/jupyter lab analyse_donnees.ipynb   # (optionnel) AED et préparation pas à pas
.venv/bin/panel serve app.py --show   # http://localhost:5006/app
```

## Publier sur GitHub Pages

```bash
./build_pages.sh   # construit docs/ (à relancer après chaque modification)
```

Puis, sur GitHub : *Settings → Pages → Deploy from a branch → main / `docs`*. La page fait tourner Python dans le
navigateur (Pyodide) : le premier chargement prend 10–30 s, ensuite tout est instantané.

## Fichiers

| Fichier | Rôle |
|---|---|
| `prepare_data.py` | Nettoyage, melt, jointure des 3 sources, imputation, regroupement KMeans |
| `analyse_donnees.ipynb` | Les mêmes étapes expliquées, avec l'AED en graphiques (exécuté, résultats inclus) |
| `app.py` | Tableau de bord (filtres, calculs, figures, mise en page) |
| `build_pages.sh` / `docs/` | Version statique pour GitHub Pages |
| `data/raw/` | Données brutes (Kaggle + API Banque mondiale) |
| `data/clean/panel.csv` | 1 ligne = 1 pays × 1 année |
| `data/clean/countries.csv` | 1 ligne = 1 pays : statistiques résumées + profil (cluster) |

## Données : 3 sources complémentaires

| Source | Apporte | Pourquoi on en a besoin |
|---|---|---|
| JD1 – Kaggle *Unemployment dataset* (Banque mondiale / OIT) | chômage | 1re variable de la question |
| JD2 – Kaggle *Global Inflation Dataset* (Banque mondiale, Ha-Kose-Ohnsorge) | inflation globale, alimentaire, énergie | 2e variable + décomposition de l'inflation |
| JD3 – API Banque mondiale `/v2/country` | région, groupe de revenu | séparer pays et agrégats ; filtres et comparaison régionale |

Jointure sur **code ISO3 + année** → 180 pays, 5 580 lignes, 5 489 paires chômage-inflation complètes.

**Les 9 colonnes du jeu final, et où chacune sert :**

| Colonne | Utilisée dans |
|---|---|
| `iso3` | clé de jointure, carte, clic → pays |
| `country` | libellés, infobulles, tableau |
| `region` | filtre global, médiane régionale (série temporelle, tableau d'alertes) |
| `income_group` | filtre global, couleur du nuage de points |
| `year` | animation de la carte, séries temporelles, décennies, année des alertes |
| `unemployment` | tous les graphiques, KPI, alertes |
| `inflation` | mesure par défaut partout |
| `food_inflation` | inflation par catégorie (onglet Pays, Q4) |
| `energy_inflation` | inflation par catégorie (onglet Pays, Q4) |

Écartées volontairement : inflation sous-jacente et prix à la production (30–35 % de manquants, n'aident à répondre à aucune question),
code FMI, `Indicator Type` (constante), `Note`, colonnes vides.

## Méthodes d'analyse (exigences des labos)

- **AED / nettoyage / imputation** (`prepare_data.py`) : encodage latin-1 corrigé, agrégats retirés via JD3,
  melt large → long, interpolation linéaire par pays des seuls trous de 2 ans ou moins, les trous plus longs restant vides (22 valeurs alimentaires, 20 énergie),
  hyperinflation (> 50 %) toujours masquée dans le tableau de bord (la ligne reste, seule la valeur est retirée).
- **Regroupement** : KMeans (k = 4) sur 5 attributs par pays — chômage médian, inflation médiane, écart-type de chacun,
  r de Phillips (inflation en log signé, standardisation). Les profils sont nommés automatiquement d'après leurs médianes.
  Silhouette affichée pour k = 2..6 lors de l'exécution.
- **Géospatial** : cartes choroplèthes (chômage et inflation par année dans *Monde* ; r par pays sur 1991–2021 dans *Le lien*).
- **Séries temporelles** : évolution par pays vs médiane régionale, chocs 1997/2008/2020, corrélation par décennie.

## Comment le tableau de bord répond aux questions

| Question | Panneau(x) |
|---|---|
| Q1 – Inflation et chômage sont-ils liés ? | Courbe de Phillips (nuage + tendance + trajectoire du pays), KPI r médian |
| Q2 – La relation diffère-t-elle selon le pays ? | Histogramme des r par pays, carte « Corrélation r » |
| Q3 – … et dans le temps ? | r médian par décennie, série temporelle du pays avec crises, carte animée |
| Q4 – L'inflation vient-elle de l'alimentation ou de l'énergie ? | Barres globale / alimentaire / énergie du pays |
| Action – Quels pays surveiller ? | Onglet Alertes : pays en stagflation chaque année (1992–2021), tableau d'une année au choix avec historique par pays, seuils réglables, export CSV ; KPI |
| Quels pays se ressemblent ? | Onglet Profils : graphique des profils (clustering), tableau des profils, pays semblables |

Constat notable : r médian ≈ −0,10 en 1991–2000 et 2001–2010, mais ≈ −0,02 en 2011–2021 (aplatissement de la courbe de Phillips).

## Mise en page et repères

Thème sombre unique (police Inter, cartes arrondies, palette validée pour fond sombre ; tous les jetons de couleur sont définis en tête de `app.py`, section « thème sombre »).

En haut, toujours visibles : la question de recherche et 4 KPI pour l'**année affichée** (curseur de la barre latérale), avec mini-courbe de tendance où l'année est repérée (jauge −1 → +1 pour r, calculé sur les 10 ans qui finissent cette année-là). Dessous, 5 onglets (une question chacun, sans défilement) :

| Onglet | Contenu |
|---|---|
| 🌍 Monde | carte (chômage ou inflation) + top 10, pour l'année affichée |
| 📉 Le lien | nuage chômage × inflation (un point par pays, ou les années des pays choisis), r par pays, r par décennie, carte du r et 10 liens de Phillips les plus forts (1991–2021) |
| 🧩 Profils | pays par profil (KMeans) + tableau qui explique chaque profil |
| 🔎 Pays | fiche des pays choisis dans le filtre Pays (un bouton par pays) : évolution vs région avec crises, inflation par catégorie |
| 🚨 Alertes | seuils réglables, pays en stagflation par année (cliquer une barre change l'année affichée), tableau de l'année avec historique, export CSV |

Une seule phrase 💡 par onglet, recalculée selon les filtres. Les filtres de la barre latérale (régions, revenu, pays)
s'appliquent à **tous** les onglets ; une ligne au-dessus des onglets rappelle toujours combien de pays sont affichés.
Un clic sur un pays (carte, classement, nuage, tableau) l'ajoute au filtre Pays et, depuis la carte, le classement ou le
tableau, ouvre sa fiche. Période fixe 1991–2021 ; inflation globale partout, alimentaire et énergie dans la fiche pays.

Couleurs d'état (couleur + icône + libellé, identiques partout) :

| | Chômage | Inflation |
|---|---|---|
| ✓ Sain (vert) | < 5 % | 0–3 % (cible des banques centrales : 2 %) |
| ! À surveiller (jaune) | 5–10 % | 3–10 % |
| ⚠ Préoccupant (rouge) | ≥ 10 % | > 10 % |
| ↓ Déflation (bleu clair) | – | < 0 % |

▲▼ rouges quand la variation va dans le mauvais sens. r est toujours en bleu et traduit en mots (aucun lien / faible / modéré / fort).

## Composantes exigées

1. **Interactivité**
   - Filtres globaux (barre latérale, tous les onglets) : régions, groupes de revenu, pays (jusqu'à 8, comparés dans *Le lien*).
   - Année affichée (curseur de la barre latérale) : indicateurs, carte, classement et alertes.
   - Filtres locaux : indicateur de la carte, pays affiché dans la fiche, seuils d'alerte et « seulement les pays
     en alerte » (onglet Alertes).
   - Liens entre graphiques : un clic sur un pays l'ajoute au filtre Pays (et ouvre sa fiche depuis la carte, le classement ou le tableau) ; un clic sur une barre d'Alertes choisit l'année.
   - Tri : classement des 10 pays, tableau triable et filtrable ; zoom et survol sur tous les graphiques.
2. **Aperçu des données** : 4 KPI avec repère et tendance (toujours visibles), fiche pays (évolution vs région), tendances (séries, décennies, alertes par année),
   comparaisons (carte, classement, région, profils).
3. **Guide d'utilisation** : question de recherche, onglets nommés, une phrase 💡 par onglet, légende des couleurs, bouton *Guide*.
4. **Action / décision** : alertes de stagflation (icône + libellé + couleur), pays semblables, écart à la région, export CSV.

Types de graphiques : carte choroplèthe, barres horizontales (classement), nuage de points, histogramme, barres,
nuage des profils (échelle log), lignes (2 panneaux), barres groupées, tableau, KPI.
