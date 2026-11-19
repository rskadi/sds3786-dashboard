"""Préparation des données : nettoyage, jointure, imputation et regroupement (clustering).

Entrées (data/raw/) :
  - unemployment analysis.csv         JD1 : chômage (% pop. active, OIT modélisé), 1991-2021
  - Global Dataset of Inflation.csv   JD2 : inflation globale / alimentaire / énergie, 1970-2022
  - countries.json                    JD3 : métadonnées Banque mondiale (région, groupe de revenu)

Sorties (data/clean/) :
  - panel.csv      1 ligne = 1 pays x 1 année (9 colonnes)
  - countries.csv  1 ligne = 1 pays, statistiques résumées + profil économique (KMeans)

Usage : python prepare_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAW, CLEAN = Path("data/raw"), Path("data/clean")
YEARS = range(1991, 2022)  # période commune aux deux jeux (JD1 s'arrête en 2021)
SERIES = {  # on garde les 3 séries d'inflation utiles ; sous-jacente et prix à la production ont 30-35 % de manquants
    "Headline Consumer Price Inflation": "inflation",
    "Food Consumer Price Inflation": "food_inflation",
    "Energy Consumer Price Inflation": "energy_inflation",
}
INFL_COLS = list(SERIES.values())
REGIONS_FR = {
    "East Asia & Pacific": "Asie de l'Est et Pacifique",
    "Europe & Central Asia": "Europe et Asie centrale",
    "Latin America & Caribbean": "Amérique latine et Caraïbes",
    "Middle East & North Africa": "Moyen-Orient et Afrique du Nord",
    "Middle East, North Africa, Afghanistan & Pakistan": "Moyen-Orient et Afrique du Nord",
    "North America": "Amérique du Nord",
    "South Asia": "Asie du Sud",
    "Sub-Saharan Africa": "Afrique subsaharienne",
}
INCOME_FR = {
    "High income": "Revenu élevé",
    "Upper middle income": "Revenu intermédiaire supérieur",
    "Lower middle income": "Revenu intermédiaire inférieur",
    "Low income": "Faible revenu",
    "Not classified": "Non classé",
}
N_CLUSTERS = 4  # k=2 a la meilleure silhouette mais ne sépare que « stable / instable » ; k=4 reste lisible


def load_unemployment():
    df = pd.read_csv(RAW / "unemployment analysis.csv", encoding="utf-8-sig")
    long = df.melt(id_vars=["Country Name", "Country Code"], var_name="year", value_name="unemployment")
    long = long.rename(columns={"Country Name": "country", "Country Code": "iso3"})
    long["year"] = long["year"].astype(int)
    return long


def load_inflation():
    # Fichier encodé en latin-1 (sinon « C�te d'Ivoire ») ; colonnes constantes/vides supprimées.
    df = pd.read_csv(RAW / "Global Dataset of Inflation.csv", encoding="latin-1")
    df = df[df["Series Name"].isin(SERIES)]
    year_cols = [str(y) for y in YEARS]
    long = df.melt(id_vars=["Country Code", "Series Name"], value_vars=year_cols, var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    wide = long.pivot_table(index=["Country Code", "year"], columns="Series Name", values="value").reset_index()
    return wide.rename(columns={"Country Code": "iso3", **SERIES})


def load_metadata():
    rows = json.loads((RAW / "countries.json").read_text())[1]
    meta = pd.DataFrame(
        {"iso3": r["id"], "region": r["region"]["value"].strip(), "income_group": r["incomeLevel"]["value"].strip()}
        for r in rows
    )
    meta = meta[meta["region"] != "Aggregates"]  # retire les agrégats (Monde, régions, groupes de revenu)
    meta["region"] = meta["region"].map(REGIONS_FR)
    meta["income_group"] = meta["income_group"].map(INCOME_FR).fillna("Non classé")
    return meta


MAX_GAP = 2  # longueur maximale (en années) d'un trou que l'on comble


def fill_short_gaps(s, max_gap=MAX_GAP):
    """Interpolation linéaire des trous d'au plus `max_gap` années, entourés de vraies valeurs.
    Un trou plus long reste entièrement vide (`interpolate(limit=2)` en remplirait les 2 premières années)."""
    missing = s.isna()
    gap_len = missing.groupby((~missing).cumsum()).transform("sum")  # taille du trou auquel appartient chaque année
    return s.mask(missing & (gap_len <= max_gap), s.interpolate(limit_area="inside"))


def impute(panel):
    """Comble, pays par pays, les trous de 2 ans ou moins ; les trous plus longs et les bords restent vides."""
    panel = panel.sort_values(["iso3", "year"]).copy()
    before = panel[INFL_COLS].isna().sum()
    panel[INFL_COLS] = panel.groupby("iso3")[INFL_COLS].transform(fill_short_gaps)
    filled = before - panel[INFL_COLS].isna().sum()
    print("Valeurs imputées par interpolation :", filled.to_dict())
    return panel


def country_profiles(panel):
    """Statistiques par pays + KMeans sur 5 attributs -> profil économique."""
    def stats(g):
        both = g.dropna(subset=["unemployment", "inflation"])
        return pd.Series({
            "unemp_median": g["unemployment"].median(),
            "unemp_std": g["unemployment"].std(),
            "infl_median": g["inflation"].median(),
            "infl_std": g["inflation"].std(),
            "phillips_r": both["unemployment"].corr(both["inflation"]) if len(both) >= 20 else np.nan,
        })

    prof = panel.groupby("iso3").apply(stats, include_groups=False).reset_index()
    prof = prof.merge(panel[["iso3", "country", "region", "income_group"]].drop_duplicates(), on="iso3")

    feats = prof[["unemp_median", "unemp_std", "infl_median", "infl_std", "phillips_r"]].copy()
    # L'inflation est très asymétrique (hyperinflations) -> échelle log signée avant standardisation
    for c in ["infl_median", "infl_std"]:
        feats[c] = np.sign(feats[c]) * np.log1p(feats[c].abs())
    ok = feats.notna().all(axis=1)
    X = StandardScaler().fit_transform(feats[ok])

    print("Silhouette selon k :", {k: round(silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)), 3)
                                   for k in range(2, 7)})
    km = KMeans(N_CLUSTERS, n_init=10, random_state=0).fit(X)
    prof.loc[ok, "cluster"] = km.labels_
    prof["profile"] = prof["cluster"].map(name_clusters(prof[ok])).fillna("Données insuffisantes")
    return prof.drop(columns="cluster")


def name_clusters(prof):
    """Nom lisible pour chaque groupe, déduit de ses médianes comparées à la médiane mondiale."""
    med = prof[["unemp_median", "infl_median", "infl_std"]].median()
    names = {}
    for k, g in prof.groupby("cluster"):
        c = g[["unemp_median", "infl_median", "infl_std"]].median()
        infl = "Inflation élevée et volatile" if c.infl_std > 2 * med.infl_std else (
            "Inflation élevée" if c.infl_median > med.infl_median else "Inflation faible")
        unemp = "chômage élevé" if c.unemp_median > med.unemp_median else "chômage faible"
        names[k] = f"{infl}, {unemp}"
    # Si deux groupes ont le même nom, on les distingue par le signe de leur courbe de Phillips
    for k, g in prof.groupby("cluster"):
        if list(names.values()).count(names[k]) > 1:
            names[k] += " (r < 0)" if g["phillips_r"].median() < 0 else " (r ≥ 0)"
    return names


def main():
    unemp, infl, meta = load_unemployment(), load_inflation(), load_metadata()
    print(f"JD1 chômage : {unemp.iso3.nunique()} entités | JD2 inflation : {infl.iso3.nunique()} pays | "
          f"JD3 métadonnées : {len(meta)} pays")

    # Jointure : pays présents dans les 3 sources (les agrégats de JD1 disparaissent via JD3)
    panel = unemp.merge(meta, on="iso3").merge(infl, on=["iso3", "year"], how="left")
    panel = panel[panel.iso3.isin(infl.iso3)]
    panel = impute(panel)

    cols = ["iso3", "country", "region", "income_group", "year", "unemployment"] + INFL_COLS
    panel = panel[cols].round(2)
    print(f"Panel final : {panel.iso3.nunique()} pays, {len(panel)} lignes, "
          f"{panel.dropna(subset=['unemployment', 'inflation']).shape[0]} paires chômage-inflation complètes")
    print("Manquants restants (%) :", (panel[INFL_COLS].isna().mean() * 100).round(1).to_dict())

    prof = country_profiles(panel)
    print("Profils :", prof.profile.value_counts().to_dict())

    CLEAN.mkdir(exist_ok=True)
    panel.to_csv(CLEAN / "panel.csv", index=False)
    prof.round(3).to_csv(CLEAN / "countries.csv", index=False)
    print("Écrit :", CLEAN / "panel.csv", CLEAN / "countries.csv")


if __name__ == "__main__":
    main()
