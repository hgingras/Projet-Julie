import pandas as pd
from ast import literal_eval
from typing import Dict



def jobids_SM_active_below_5(df: pd.DataFrame) -> Dict[int, int]:
    """
    Input dataframe ressources utilisées.
    
    Retourne un dict {slurmjobid: 5} pour les jobs dont *toutes* les lignes
    ont gpu_util < 5.

    Suppose que les colonnes 'slurmjobid' et 'gpu_util' existent.
    Ne modifie pas df.
    """
    # Option : si tu veux ignorer les NaN dans le test, décommente la ligne suivante :
    # df = df.copy()
    # df["gpu_util"] = df["gpu_util"].fillna(np.inf)  # NaN -> échec du test (<5) car inf < 5 est False

    valid = (
        df.groupby("slurmjobid")["gpu_util"]
          .apply(lambda s: (s < 5).all())
    )

    # slurmjobid valides
    jobids = valid[valid].index

    # Construire {slurmjobid: 5}
    return {int(j): 5 for j in jobids}


# pas besoin ... à voir

def load_jobs_file(path):
    """Charge un fichier jobs et reconvertit la colonne codes en listes."""
    df = pd.read_excel(path)

    if "code" in df.columns:
        df["code"] = df["codes"].apply(
            lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
    return df


def classify_jobs_mig(df_gpu: pd.DataFrame, jobids: list[int]) -> dict[int, int]:
    """
    *** rouler get_a100_jobs_over_600s avant ***
    
    Renvoie {JobID: code} selon:
      - code 6 : 0 < gpu_max < 10  et mem_max <= 5
      - code 7 : 10 <= gpu_max < 20 et mem_max <= 10
      - code 8 : 20 <= gpu_max < 40 et mem_max <= 20
    On ignore un job seulement si TOUTES ses mesures gpu_util == 0.
    df_gpu contient les colonnes: slurmjobid (ou JobID), gpu_util, memory_util.
    """
    # Normaliser les noms au besoin
    cols = {c.lower(): c for c in df_gpu.columns}
    jid_col = cols.get("slurmjobid") or cols.get("jobid")
    if jid_col is None:
        raise KeyError(f"Colonne slurmjobid/JobID introuvable: {df_gpu.columns.tolist()}")
    gu_col = cols.get("gpu_util")
    mu_col = cols.get("memory_util")
    if gu_col is None or mu_col is None:
        raise KeyError("Colonnes gpu_util et/ou memory_util introuvables")

    out: dict[int, int] = {}

    for job in jobids:
        sub = df_gpu[df_gpu[jid_col].astype('int64') == int(job)]
        if sub.empty:
            continue

        gpu_vals = sub[gu_col].to_numpy()
        mem_max  = float(sub[mu_col].max())

        # ▶️ ignorer seulement si le job est entièrement inactif
        if (gpu_vals == 0).all():
            continue

        gpu_max = float(gpu_vals.max())

        # Codes MIG
        if (5 < gpu_max < 10)  and (mem_max <= 5):
            out[int(job)] = 6
        elif (10 <= gpu_max < 20) and (mem_max <= 10):
            out[int(job)] = 7
        elif (20 <= gpu_max < 40) and (mem_max <= 20):
            out[int(job)] = 8
        elif (5 < gpu_max < 40) and (mem_max <= 20):
            out[int(job)] = 8
        # sinon, le job ne rentre pas dans ces profils → pas ajouté

    return out