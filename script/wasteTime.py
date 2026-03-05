from typing import Dict, Optional, Union
import pandas as pd

def jobids_under_10_minutes(df: pd.DataFrame) -> Dict[int, int]:
    """
    Input dataframe ressources demandées.
    
    Retourne un dict {JobID: 1} pour les lignes où Elapsed < 600 s (inclut 0).
    Ne modifie pas df et n'ajoute rien à la colonne 'codes'.
    Suppose que les colonnes 'JobID' et 'Elapsed' existent.
    """
    # Masque: Elapsed < 600 et valeurs non nulles
    mask = df["Elapsed"].notna() & (df["Elapsed"] < 600) & df["JobID"].notna()

    # Récupérer les JobID, les convertir en int (élimine aussi les NaN résiduels)
    jobids = df.loc[mask, "JobID"].astype(int)

    # Construire {JobID: 1}
    return {int(j): 1 for j in jobids.tolist()}

def _to_datetime_from_excel(col: pd.Series) -> pd.Series:
    """
    Convertit une colonne de timestamps en datetime si besoin.
    - Si déjà datetime -> renvoie tel quel
    - Si numérique (Excel serial days) -> convertit avec origin '1899-12-30'
    - Sinon tente pd.to_datetime (coerce)
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        # Excel serial days -> pandas datetime (UTC-naive)
        return pd.to_datetime(col, unit="D", origin="1899-12-30")
    # Sinon, on tente une conversion "classique"
    return pd.to_datetime(col, errors="coerce")

def jobids_with_trailing_idle(
    df: pd.DataFrame,
    *,
    minutes_idle: int = 30,
    jobid_col: str = "slurmjobid",
    ts_col: str = "timestamp_excel",
    util_col: str = "gpu_util",
) -> Dict[int, int]:
    """
    Renvoie un dict {JobID: 4} pour les jobs ayant :
      - au moins UNE mesure util > 0,
      - puis, uniquement des util == 0 sur >= `minutes_idle` jusqu'à la fin du job.

    Hypothèses :
    - Un job apparaît sur plusieurs lignes (une ligne = un pas de 10 min chez toi),
      mais on ne suppose pas que l'intervalle est STRICTEMENT constant : on mesure
      la durée réelle avec les timestamps.
    - Le DataFrame `df` contient au minimum (jobid_col, ts_col, util_col).

    Paramètres :
    - minutes_idle : seuil de "queue idle" (par défaut 30 minutes).
    - jobid_col/ts_col/util_col : noms de colonnes (par défaut adaptés au fichier GPU).
    """
    # Garde-fous de colonnes
    missing = [c for c in (jobid_col, ts_col, util_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}")

    # Timestamp en datetime si besoin
    ts = _to_datetime_from_excel(df[ts_col])
    df = df.copy()
    df[ts_col] = ts

    # On jette les lignes sans timestamp interprétable
    df = df.dropna(subset=[ts_col])

    out: Dict[int, int] = {}

    # Groupement par JobID
    for job, g in df.groupby(jobid_col, sort=False):
        g = g.sort_values(ts_col)

        util = g[util_col].fillna(0)

        # 1) Le job doit avoir AU MOINS une mesure > 0 (activité observée)
        has_activity = (util > 0).any()
        if not has_activity:
            continue

        # 2) Trouver la DERNIÈRE mesure > 0
        last_active_idx = util[util > 0].index.max()

        # 3) "Queue" = toutes les lignes APRÈS la dernière activité
        tail = g.loc[g.index > last_active_idx]

        # S'il n'y a pas de ligne après, pas de queue -> on ignore
        if tail.empty:
            continue

        # 4) Toutes les mesures de la queue doivent être == 0
        if (tail[util_col].fillna(0) != 0).any():
            # Il y a encore de l'activité après la "dernière" : en fait l'activité n'était pas la dernière
            # ou il y a eu reprise -> ne correspond pas au pattern souhaité
            continue

        # 5) Durée de la queue idle : de la dernière activité au dernier timestamp du job
        last_active_time = g.loc[last_active_idx, ts_col]
        last_time = g[ts_col].iloc[-1]
        idle_minutes = (last_time - last_active_time).total_seconds() / 60.0

        # 6) Vérifier le seuil (>= minutes_idle)
        if idle_minutes >= minutes_idle:
            try:
                out[int(job)] = 4
            except Exception:
                # Si job n'est pas castable en int, on le garde tel quel
                out[job] = 4

    return out

def get_a100_jobs_over_600s(df: pd.DataFrame) -> list[int]:
    """
    Input dataframe ressources demandées.
    
    Retourne la liste des JobID qui :
    - ont demandé un GPU A100
    - ont duré plus de 600 secondes
    """
    mask = (
        (df["gpu_type"] == "gres/gpu:a100") &
        (df["Elapsed"] > 600)
    )
    jobids = df.loc[mask, "JobID"].unique().tolist()
    return jobids

