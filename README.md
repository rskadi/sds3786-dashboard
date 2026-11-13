# Inflation et chômage dans le monde (1991–2021)

Tableau de bord interactif (Panel + Plotly) — SDS3786, Automne 2026.
Rayane Slimane-Kadi · Mohamed Elsobky

**Question centrale :** existe-t-il une relation entre l'inflation et le chômage (courbe de Phillips),
et cette relation diffère-t-elle selon les pays et dans le temps ?

## Lancer

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_data.py      # data/raw -> data/clean
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
| `year` | filtre de période, animation de la carte, séries temporelles, décennies |
| `unemployment` | tous les graphiques, KPI, alertes |
| `inflation` | mesure par défaut partout |
| `food_inflation` | sélecteur « type d'inflation », décomposition Q4 |
| `energy_inflation` | sélecteur « type d'inflation », décomposition Q4 |

Écartées volontairement : inflation sous-jacente et prix à la production (30–35 % de manquants, n'aident à répondre à aucune question),
code FMI, `Indicator Type` (constante), `Note`, colonnes vides.

## Méthodes d'analyse (exigences des labos)

- **AED / nettoyage / imputation** (`prepare_data.py`) : encodage latin-1 corrigé, agrégats retirés via JD3,
  melt large → long, interpolation linéaire des trous ≤ 2 ans par pays (46 valeurs alimentaires, 94 énergie),
  hyperinflation (> 50 %) masquée par défaut dans le tableau de bord (désactivable).
- **Regroupement** : KMeans (k = 4) sur 5 attributs par pays — chômage médian, inflation médiane, écart-type de chacun,
  r de Phillips (inflation en log signé, standardisation). Les profils sont nommés automatiquement d'après leurs médianes.
  Silhouette affichée pour k = 2..6 lors de l'exécution.
- **Géospatial** : carte choroplèthe (chômage, inflation, r par pays, profil).
- **Séries temporelles** : évolution par pays vs médiane régionale, chocs 1997/2008/2020, corrélation par décennie.

## Comment le tableau de bord répond aux questions

| Question | Panneau(x) |
|---|---|
| Q1 – Inflation et chômage sont-ils liés ? | Courbe de Phillips (nuage + tendance + trajectoire du pays), KPI r médian |
| Q2 – La relation diffère-t-elle selon le pays ? | Histogramme des r par pays, carte « Corrélation r » |
| Q3 – … et dans le temps ? | r médian par décennie, série temporelle du pays avec crises, carte animée |
| Q4 – L'inflation vient-elle de l'alimentation ou de l'énergie ? | Barres globale / alimentaire / énergie du pays |
| Action – Quels pays surveiller ? | Tableau d'alertes de stagflation (seuils réglables, tri, export CSV), KPI |
| Quels pays se ressemblent ? | Graphique des profils (clustering), filtre global « profil » |

Constat notable : r médian ≈ −0,10 en 1991–2000 et 2001–2010, mais ≈ −0,02 en 2011–2021 (aplatissement de la courbe de Phillips).

## Mise en page et repères

Thème sombre unique (police Inter, cartes arrondies, palette validée pour fond sombre ; tous les jetons de couleur sont définis en tête de `app.py`, section « thème sombre »).

En haut, toujours visibles : la question de recherche et 4 KPI avec mini-courbe de tendance (jauge −1 → +1 pour r). Dessous, 5 onglets (une question chacun, sans défilement) :

| Onglet | Contenu |
|---|---|
| 🌍 Monde | carte (chômage, inflation, r ou profil ; ▶ anime les années) + top 10 |
| 📉 Le lien | nuage chômage × inflation avec trajectoire du pays, histogramme des r par pays, r par décennie |
| 🧩 Profils | pays par profil (KMeans) + tableau qui explique chaque profil |
| 🔎 Pays | fiche : 4 KPI vs région, évolution avec crises, inflation par catégorie |
| 🚨 Alertes | tableau triable/filtrable, export CSV |

Une seule phrase 💡 par onglet, recalculée selon les filtres. Un clic sur un pays (carte, classement, tableau) ouvre
sa fiche ; dans les nuages de points, un clic le surligne sur place.

Couleurs d'état (couleur + icône + libellé, identiques partout) :

| | Chômage | Inflation |
|---|---|---|
| ✓ Sain (vert) | < 5 % | 0–3 % (cible des banques centrales : 2 %) |
| ! À surveiller (jaune) | 5–10 % | 3 % → seuil d'alerte |
| ⚠ Préoccupant (rouge) | ≥ 10 % | > seuil (10 % par défaut, réglable) |
| ↓ Déflation (bleu clair) | – | < 0 % |

▲▼ rouges quand la variation va dans le mauvais sens. r est toujours en bleu et traduit en mots (aucun lien / faible / modéré / fort).

## Composantes exigées

1. **Interactivité**
   - Filtres globaux (barre latérale) : période, type d'inflation, hyperinflation, région, revenu, profil, seuils d'alerte.
   - Filtres locaux : indicateur et année de la carte (lecteur ▶), couleur et niveau de détail du nuage, comparaison
     régionale, « seulement les pays en alerte ».
   - Liens entre graphiques : un clic sur la carte, le classement ou le tableau ouvre la fiche du pays ; dans les nuages, il le surligne.
   - Tri : classement des 10 pays, tableau triable et filtrable ; zoom et survol sur tous les graphiques.
2. **Aperçu des données** : 4 KPI avec repère et tendance (toujours visibles), fiche pays (4 KPI vs région), tendances (séries, décennies),
   comparaisons (carte, classement, région, profils).
3. **Guide d'utilisation** : question de recherche, onglets nommés, une phrase 💡 par onglet, légende des couleurs, bouton *Guide*.
4. **Action / décision** : alertes de stagflation (icône + libellé + couleur), pays semblables, écart à la région, export CSV.

Types de graphiques : carte choroplèthe, barres horizontales (classement), nuage de points, histogramme, barres,
nuage des profils (échelle log), lignes (2 panneaux), barres groupées, tableau, KPI.
