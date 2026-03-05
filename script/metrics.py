import pandas as pd
import numpy as np

def export_gpu_memory_metrics(
    prom,
    account_query: str,
    prom_filter: str,
    d_from,
    d_to,
    step,
    output_file: str = "gpu_metrics.xlsx",
    convert_memory_to_gib: bool = True,
) -> pd.DataFrame:
    """
    Exécute deux requêtes Prometheus (utilisation GPU et mémoire associée),
    consolide les résultats, pivote les données pour avoir les deux métriques
    côte à côte, et exporte en Excel.

    Paramètres
    ----------
    prom : objet client Prometheus
        Doit exposer la méthode `custom_query_range(query, start_time, end_time, step)`.
    account_query : str
        Valeur du label 'account' à injecter dans les requêtes.
    prom_filter : str
        Filtre Prometheus additionnel (ex: 'cluster="prod",job="dcgm"' ou tout autre labels selector).
        Ne pas inclure d'accolades.
    d_from : datetime
        Début de la fenêtre temporelle (timezone-aware recommandé).
    d_to : datetime
        Fin de la fenêtre temporelle (timezone-aware recommandé).
    step : str | int | float
        Pas de requête (ex: "60s", "5m", 60).
    output_file : str, optionnel
        Nom du fichier Excel de sortie. Par défaut "gpu_metrics.xlsx".
    convert_memory_to_gib : bool, optionnel
        Si True, convertit la mémoire de bytes en GiB.

    Retour
    ------
    pd.DataFrame
        DataFrame large (pivoté) comprenant les colonnes :
        ['slurmjobid', 'instance', 'timestamp', 'gpu_util', 'memory_util', 'timestamp_excel'].
        Note : 'timestamp' est tz-aware (UTC), 'timestamp_excel' est naïf (sans timezone).

    Effets
    ------
    - Écrit un fichier Excel avec une feuille 'metrics' et les colonnes :
      ['slurmjobid', 'instance', 'timestamp_excel', 'gpu_util', 'memory_util'].
    """
    BYTES_PER_GIB = 1024 ** 3

    # Préparation des requêtes
    query_dict = {
        "gpu_util":    f'slurm_job_utilization_gpu{{account="{account_query}", {prom_filter}}}',
        "memory_util": f'slurm_job_memory_usage_gpu{{account="{account_query}", {prom_filter}}}',
    }

    rows = []

    for label, query in query_dict.items():
        result = prom.custom_query_range(
            query=query,
            start_time=d_from,
            end_time=d_to,
            step=step
        )

        for entry in result:
            metric_info = entry.get("metric", {})
            slurmjobid  = metric_info.get("slurmjobid", "N/A")
            metric_name = metric_info.get("__name__", "N/A")
            instance    = metric_info.get("instance", "N/A")

            for ts, val in entry.get("values", []):
                # Prometheus -> epoch SECONDS
                ts_dt = pd.to_datetime(float(ts), unit="s", utc=True)

                try:
                    v = float(val)
                except (ValueError, TypeError):
                    v = np.nan

                # Conversion optionnelle: bytes -> GiB pour memory_util
                if label == "memory_util" and convert_memory_to_gib and pd.notna(v):
                    v = v / BYTES_PER_GIB

                rows.append({
                    "slurmjobid": slurmjobid,
                    "metric_name": metric_name,   # informatif
                    "instance": instance,
                    "timestamp": ts_dt,           # tz-aware UTC
                    "value": v,                   # gpu_util: ratio 0..1 | memory_util: octets ou GiB
                    "label": label                # 'gpu_util' ou 'memory_util'
                })

    # --- DataFrame long ---
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        print("No data returned.")
        # Créer un DF vide avec la bonne structure pour cohérence
        df_wide = pd.DataFrame(columns=["slurmjobid", "instance", "timestamp", "gpu_util", "memory_util", "timestamp_excel"])
        # Écrit tout de même un fichier Excel vide/squelette si souhaité
        with pd.ExcelWriter(output_file, engine="openpyxl",
                            datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
            df_wide[["slurmjobid", "instance", "timestamp_excel", "gpu_util", "memory_util"]].to_excel(
                writer, sheet_name="metrics", index=False
            )
        print(f"Wrote → {output_file} (empty dataset)")
        return df_wide

    # Sécuriser le type temps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # --- PIVOT pour avoir 2 colonnes côte à côte ---
    # Une ligne par (slurmjobid, instance, timestamp), une colonne par label
    df_wide = (
        df.pivot_table(
            index=["slurmjobid", "instance", "timestamp"],
            columns="label",
            values="value",
            aggfunc="mean"   # en cas de doublons (rare), on prend la moyenne
        )
        .reset_index()
    )

    # S'assurer que les colonnes existent même si une requête ne retourne rien
    for needed in ["gpu_util", "memory_util"]:
        if needed not in df_wide.columns:
            df_wide[needed] = np.nan

    # Colonne Excel-friendly (sans timezone)
    df_wide["timestamp_excel"] = (
        df_wide["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    )

    # Colonnes exportées (gpu_util et memory_util côte à côte)
    ordered_cols = [
        "slurmjobid", "instance", "timestamp_excel",
        "gpu_util", "memory_util"
    ]

    # Export (répertoire courant)
    with pd.ExcelWriter(output_file, engine="openpyxl",
                        datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        df_wide[ordered_cols].to_excel(writer, sheet_name="metrics", index=False)

    print(f"Wrote → {output_file}")
    return df_wide



def adjust_columns_get_records(
    df: pd.DataFrame
) -> pd.DataFrame:

    jobs_frame = df.copy()
    
    jobs_frame['ElapsedHours'] = jobs_frame['Elapsed'] / 3600 # Scale to hours
    jobs_frame['WaitTimes'] = (jobs_frame['Start'] - jobs_frame['Submit']).dt.total_seconds().divide(3600)
    jobs_frame['TimelimitDelta'] = pd.to_timedelta(jobs_frame['Timelimit'], unit='m')
    
    jobs_frame['eligible_wait'] = jobs_frame['Start'] - jobs_frame['Eligible']
    jobs_frame['eligible_wait_sec']=jobs_frame['eligible_wait'] / np.timedelta64(1, 's')
    jobs_frame['eligible_wait_time']=pd.to_timedelta(jobs_frame['eligible_wait_sec'], unit='s')
    jobs_frame['eligible_wait_hours']=jobs_frame['eligible_wait_sec']/3600
    
    jobs_frame['eligible_delta'] = jobs_frame['Eligible'] - jobs_frame['Submit']
    jobs_frame['eligible_delta_sec']=jobs_frame['eligible_delta'] / np.timedelta64(1, 's')
    jobs_frame['eligible_delta_time']=pd.to_timedelta(jobs_frame['eligible_delta_sec'], unit='s')
    jobs_frame['eligible_delta_hours']=jobs_frame['eligible_delta_sec']/3600
    
    jobs_frame['timelimit_hours'] = jobs_frame['Timelimit'] / 60
    
    jobs_frame['time_delta_hours'] = jobs_frame['timelimit_hours'] - jobs_frame['ElapsedHours']
    jobs_frame['time_delta_norm'] = jobs_frame['time_delta_hours'] / jobs_frame['timelimit_hours']
    
    jobs_frame['billing'] = (
        jobs_frame['AllocTRES']
            .astype('string')
            .str.extract(r'billing=(\d+)', expand=False)
            .astype('Int64'))
    
    jobs_frame['cpu'] = (
        jobs_frame['AllocTRES']
            .astype('string')
            .str.extract(r'cpu=(\d+)', expand=False)
            .astype('Int64'))
    
    jobs_frame['gpu_type'] = (
        jobs_frame['AllocTRES']
            .astype("string")
            .str.extract(r'(gres\/[^=]+)', expand=False))
    
    jobs_frame['gpu'] = (
        jobs_frame['AllocTRES']
            .astype('string')
            .str.extract(r'gpu=(\d+)', expand=False)
            .astype('Int64'))
    
    # Extraire la valeur et l'unité ; si mem n'existe pas → NaN
    mem_extract = (
        jobs_frame['AllocTRES']
        .astype('string')
        .str.extract(r'mem=(\d+)([MGmg])', expand=True)
        .rename(columns={0: 'mem_val', 1: 'mem_unit'})
    )
    
    # Conversion de mem_val en float (nullable)
    mem_val = mem_extract['mem_val'].astype('Float64')
    
    # Facteur de conversion : M → 1024, G → 1
    factor = mem_extract['mem_unit'].str.upper().map({'M': 1024.0, 'G': 1.0})
    
    # Conversion en GiB, sans arrondi
    # where() garantit que si mem est absent → NaN (donc case vide)
    jobs_frame['memory_Gb'] = (mem_val / factor).where(
        mem_extract['mem_val'].notna() & mem_extract['mem_unit'].notna()
    )

    jobs_frame['GPU_CPU_day'] = jobs_frame['Elapsed'] * jobs_frame['billing'] /86400000
    jobs_frame = jobs_frame.sort_values(
        by=['Account', 'Start'],
        ascending=[True, True],
        na_position='last'
    ).reset_index(drop=True)
    
    jobs_frame['code'] = ""

    return jobs_frame


def export_cpu_memory_metrics(
    prom,
    account_query: str,
    prom_filter: str,
    d_from,
    d_to,
    step,
    output_file: str = "cpu_metrics.xlsx",
    convert_memory_to_gib: bool = True,
) -> pd.DataFrame:
    """
    Exécute deux requêtes Prometheus (utilisation GPU et mémoire associée),
    consolide les résultats, pivote les données pour avoir les deux métriques
    côte à côte, et exporte en Excel.

    Paramètres
    ----------
    prom : objet client Prometheus
        Doit exposer la méthode `custom_query_range(query, start_time, end_time, step)`.
    account_query : str
        Valeur du label 'account' à injecter dans les requêtes.
    prom_filter : str
        Filtre Prometheus additionnel (ex: 'cluster="prod",job="dcgm"' ou tout autre labels selector).
        Ne pas inclure d'accolades.
    d_from : datetime
        Début de la fenêtre temporelle (timezone-aware recommandé).
    d_to : datetime
        Fin de la fenêtre temporelle (timezone-aware recommandé).
    step : str | int | float
        Pas de requête (ex: "60s", "5m", 60).
    output_file : str, optionnel
        Nom du fichier Excel de sortie. Par défaut "gpu_metrics.xlsx".
    convert_memory_to_gib : bool, optionnel
        Si True, convertit la mémoire de bytes en GiB.

    
    Retour
    ------
    pd.DataFrame
        DataFrame large (pivoté) comprenant les colonnes :
        ['slurmjobid', 'instance', 'timestamp', 'gpu_util', 'memory_util', 'timestamp_excel'].
        Note : 'timestamp' est tz-aware (UTC), 'timestamp_excel' est naïf (sans timezone).

    Effets
    ------
    - Écrit un fichier Excel avec une feuille 'metrics' et les colonnes :
      ['slurmjobid', 'instance', 'timestamp_excel', 'gpu_util', 'memory_util'].
    """
    
    BYTES_PER_GIB = 1024 ** 3

    # Préparation des requêtes

    query_dict = {
    "cpu_util": (
        f'avg by (slurmjobid) ('
        f'  rate(slurm_job_core_usage_total{{account="{account_query}", {prom_filter}}}[10m])'
        f')/1000000000'
    ),
    }
    df = pd.DataFrame()
    for label, query in query_dict.items():
        result = prom.custom_query_range(query=query, start_time=d_from, end_time=d_to, step=step)
        part = MetricRangeDataFrame(result).rename(columns={"value": label})
        df = pd.concat([df, part], axis=1)
        
    df = df.sort_index()

    query_dict = {
    "memory_cpu_util": (
        f'(slurm_job_memory_max{{account="{account_query}", {prom_filter}}})/{BYTES_PER_GIB}'
    ),
    }

    # correction du bug dans la boucle df2
    df2 = pd.DataFrame()
    for label, query in query_dict.items():
        result = prom.custom_query_range(query=query, start_time=d_from, end_time=d_to, step=step)
        part = MetricRangeDataFrame(result).rename(columns={"value": label})
        df2 = pd.concat([df2, part], axis=1)

    df2 = df2.sort_index()

    # 1) S'assurer que 'timestamp' est en colonne puis composer un MultiIndex commun
    cpu = df.reset_index()    # -> timestamp devient colonne
    mem = df2.reset_index()

    # 2) Harmoniser le type de slurmjobid (au choix: str ou int, mais identique des 2 côtés)
    cpu['slurmjobid'] = cpu['slurmjobid'].astype(str)
    mem['slurmjobid'] = mem['slurmjobid'].astype(str)

    # 3) (Optionnel) Harmoniser timezone / arrondir si nécessaire
    cpu['timestamp'] = pd.to_datetime(cpu['timestamp']).dt.floor('10min')  # uniquement si tu as des pas qui diffèrent
    mem['timestamp'] = pd.to_datetime(mem['timestamp']).dt.floor('10min')  # adapte si besoin

    # 4) Réduire mem aux colonnes utiles pour éviter les collisions
    mem_narrow = mem[['timestamp', 'slurmjobid', 'memory_cpu_util']]

    # 5) Fusion explicite (inner si tu veux seulement les paires communes)
    df_final = (
        cpu.merge(mem_narrow, on=['timestamp','slurmjobid'], how='outer')
           .set_index(['timestamp','slurmjobid'])
           .sort_index()
)   
    return df_final