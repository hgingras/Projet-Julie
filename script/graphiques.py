from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ast
import re

def build_levelfs_with_jobs_chart(
    sshare_frame: pd.DataFrame,
    jobs_frame: pd.DataFrame,
    account_query: str,
    system: str,
    *,
    y_max: float = 0.05,
    max_job_marker_px: int = 36,
    show: bool = False,
) -> go.Figure:
    """
    Construit une figure Plotly combinant :
      1) la courbe LevelFS dans le temps (filtrée par Account == account_query et Cluster == system),
      2) la superposition de points représentant les jobs (taille ∝ billing) sur un axe Y secondaire "muet".

    Paramètres
    ----------
    sshare_frame : pd.DataFrame
        Doit contenir au minimum les colonnes ['Account', 'Cluster', 'date', 'LevelFS'].
        (Tu as déjà fait `sshare_frame = sshare_frame.reset_index()` en amont.)
    jobs_frame : pd.DataFrame
        Doit contenir au minimum ['JobID', 'Start', 'billing'].
    account_query : str
        Le compte (ex: "def-jfaure_gpu").
    system : str
        Le cluster (ex: "narval"). Utilisé à la place d'une variable `cluster`.
    y_max : float, optionnel
        Borne supérieure de l'axe Y pour LevelFS (défaut: 0.05).
    max_job_marker_px : int, optionnel
        Taille visuelle maximale (en px) des marqueurs des jobs (défaut: 36).
    show : bool, optionnel
        Si True, appelle `fig.show()` avant de retourner la figure.

    Retour
    ------
    go.Figure
        La figure Plotly prête à être affichée.
    """
    # --------- Vérifications minimales des colonnes attendues ---------
    required_sshare_cols = {"Account", "Cluster", "date", "LevelFS"}
    missing_sshare = required_sshare_cols - set(sshare_frame.columns)
    if missing_sshare:
        raise ValueError(f"Colonnes manquantes dans sshare_frame: {missing_sshare}")

    required_jobs_cols = {"JobID", "Start", "billing"}
    missing_jobs = required_jobs_cols - set(jobs_frame.columns)
    if missing_jobs:
        raise ValueError(f"Colonnes manquantes dans jobs_frame: {missing_jobs}")

    # =========================
    # 1) Courbe LevelFS
    # =========================
    # Filtrage par compte et cluster (system)
    df = sshare_frame[
        (sshare_frame["Account"] == account_query) & (sshare_frame["Cluster"] == system)
    ].copy()

    # Harmonisation/tri dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    # Nettoyage des valeurs infinies
    df["LevelFS"] = df["LevelFS"].replace([np.inf, -np.inf], np.nan)

    # Courbe principale
    fig = px.line(
        df,
        x="date",
        y="LevelFS",
        title=f"LevelFS dans le temps — {account_query} ({system})",
        labels={"date": "Date", "LevelFS": "LevelFS"},
    )

    # Fixe l'axe Y : [0, y_max]
    fig.update_yaxes(range=[0, y_max])

    # Points > y_max (clampés visuellement à y_max) + annotation de la valeur réelle
    df_over = df[(df["LevelFS"].notna()) & (df["LevelFS"] > y_max)].copy()
    if not df_over.empty:
        df_over["txt"] = df_over["LevelFS"].map(lambda v: f"{v:.4f}")
        fig.add_scatter(
            x=df_over["date"],
            y=[y_max] * len(df_over),
            mode="markers+text",
            marker=dict(color="crimson", size=8, symbol="triangle-up"),
            text=df_over["txt"],
            textposition="top center",
            textfont=dict(color="crimson"),
            name=f"> {y_max:.2f} (valeur réelle)",
            hovertemplate=(
                "Date: %{x}<br>"
                "LevelFS réel: %{customdata:.6f}<extra></extra>"
            ),
            customdata=df_over["LevelFS"].values,
            showlegend=True,
        )

    # Ligne de seuil à y = y_max
    fig.add_hline(
        y=y_max,
        line_color="gray",
        line_dash="dash",
        annotation_text=f"Seuil = {str(y_max).replace('.', ',')}",
        annotation_position="top left",
    )

    # =========================
    # 2) Superposition des jobs (points ∝ billing)
    # =========================
    jobs = jobs_frame[["JobID", "Start", "billing"]].copy()
    jobs = jobs[jobs["billing"].notna()]
    jobs["Start"] = pd.to_datetime(jobs["Start"], errors="coerce")
    jobs = jobs.dropna(subset=["Start"])

    if not jobs.empty:
        # Échelle de taille en 'area' (recommandation Plotly)
        sizeref = 2.0 * jobs["billing"].max() / (max_job_marker_px**2)

        # Axe Y secondaire (muet) superposé
        fig.update_layout(
            yaxis2=dict(
                title=None,
                overlaying="y",
                side="right",
                range=[0, 1],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            )
        )

        # Points des jobs (affichés sur l'axe y2, à y = 0.5)
        fig.add_trace(
            go.Scatter(
                x=jobs["Start"],
                y=[0.5] * len(jobs),
                mode="markers",
                marker=dict(
                    size=jobs["billing"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=4,
                    color="royalblue",
                    opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                name="Jobs (taille ~ billing)",
                customdata=jobs["JobID"].astype(str).values,
                hovertemplate="JobID: %{customdata}<extra></extra>",
                yaxis="y2",
                showlegend=True,
            )
        )

    # =========================
    # 3) Cosmétiques finaux
    # =========================
    fig.update_layout(template="plotly_white", hovermode="x unified")

    if show:
        fig.show()

    return fig



def _to_datetime_safely(s: pd.Series) -> pd.Series:
    """
    Convertit une série en datetime.
    - Si déjà datetime → renvoie tel quel.
    - Si numérique et semble être un 'Excel serial date' → origin='1899-12-30'.
    - Sinon, pd.to_datetime(errors='coerce').
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    if pd.api.types.is_numeric_dtype(s):
        non_na = s.dropna()
        # Heuristique: des 'Excel serial dates' sont souvent > 10_000
        if len(non_na) and (non_na.gt(10_000).mean() > 0.9):
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

    converted = pd.to_datetime(s, errors="coerce")
    if converted.isna().all() and pd.api.types.is_numeric_dtype(s):
        converted = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return converted


def plot_job_durations_by_start(
    df: pd.DataFrame,
    *,
    color_scale: str = "Viridis",
    title: Optional[str] = "Durée des jobs (minutes) — colorée par billing",
    sort_by_start: bool = True,
    datetime_fmt: str = "%Y-%m-%d %H:%M:%S",
    # --- Nouveaux paramètres ---
    bar_width_mode: str = "auto",     # "auto" | "fixed_ms" | "category" | "none"
    fixed_bar_width: Optional[int] = None,  # en millisecondes si "fixed_ms"
    aggregate: Optional[str] = None,  # None | "hour" | "day" | "week"
    billing_agg: str = "mean",        # "mean" | "median" | "sum" pour l'agrégation
    color_cmin: Optional[float] = None,
    color_cmax: Optional[float] = None,
    show: bool = True,
):
    """
    Bar chart Plotly :
      - X = Start (dates)
      - Y = Durée (minutes) = End - Start
      - Couleur continue = billing
      - Hover : JobID, billing, Start, End, Durée (min) (via customdata + hovertemplate)

    Nouveautés :
      - bar_width_mode : contrôle la largeur des barres (auto, fixe en ms, catégorie, aucun)
      - aggregate : agrégation optionnelle (heure, jour, semaine) pour lisibilité
      - billing_agg : agrégateur pour la couleur en mode agrégé (mean, median, sum)
      - color_cmin/cmax : bornes manuelles de l'échelle de couleur
    """
    # --- Vérifs
    required = {"JobID", "Start", "End", "billing"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df: {missing}")

    # S'assurer qu'on travaille bien avec un DataFrame pandas "classique"
    data = pd.DataFrame(df).copy()

    # --- Conversions & nettoyage
    data["Start"] = _to_datetime_safely(data["Start"])
    data["End"] = _to_datetime_safely(data["End"])
    data["billing"] = pd.to_numeric(data["billing"], errors="coerce")

    data = data.dropna(subset=["Start", "End", "billing"])

    # Durée en minutes
    data["duration_min"] = (data["End"] - data["Start"]).dt.total_seconds() / 60.0
    data = data[(data["duration_min"].notna()) & (data["duration_min"] >= 0)]

    if sort_by_start:
        data = data.sort_values("Start").reset_index(drop=True)

    # --- Préparation du DataFrame final à tracer (agrégation optionnelle)
    plot_df = data.copy()
    aggregated = False
    if aggregate is not None:
        agg = aggregate.lower()
        if agg not in {"hour", "day", "week"}:
            raise ValueError("aggregate doit être None, 'hour', 'day' ou 'week'.")
        freq_map = {"hour": "H", "day": "D", "week": "W-MON"}
        freq = freq_map[agg]

        if billing_agg not in {"mean", "median", "sum"}:
            raise ValueError("billing_agg doit être 'mean', 'median' ou 'sum'.")

        # agrégation
        gb = (
            data.assign(Start_bucket=data["Start"].dt.floor(freq))
            .groupby("Start_bucket", as_index=False)
            .agg(
                duration_min=("duration_min", "sum"),
                billing=("billing", billing_agg),
                count_jobs=("JobID", "count"),
                first_start=("Start", "min"),
                last_end=("End", "max"),
            )
            .rename(columns={"Start_bucket": "Start"})
        )
        plot_df = gb
        aggregated = True

    # --- Préparer customdata + hover selon le mode (agrégé vs non agrégé)
    if aggregated:
        # Dates formatées
        start_txt = plot_df["Start"].dt.strftime(datetime_fmt).astype(object).values
        first_txt = plot_df["first_start"].dt.strftime(datetime_fmt).astype(object).values
        last_txt = plot_df["last_end"].dt.strftime(datetime_fmt).astype(object).values

        customdata = np.stack(
            [
                start_txt,                              # [0] Période (début du bucket)
                plot_df["billing"].values.astype(float),# [1] billing agrégé
                plot_df["count_jobs"].values.astype(int),# [2] nombre de jobs
                first_txt,                              # [3] plus tôt dans le bucket
                last_txt,                               # [4] plus tard dans le bucket
            ],
            axis=-1,
        )

        hovertemplate = (
            "<b>Période</b>: %{customdata[0]}<br>"
            f"<b>Billing ({billing_agg})</b>: %{customdata[1]:,.0f}<br>"
            "<b># Jobs</b>: %{customdata[2]}<br>"
            "<b>De</b>: %{customdata[3]} <b>à</b> %{customdata[4]}<br>"
            "<b>Durée totale</b>: %{y:.1f} min"
            "<extra></extra>"
        )
    else:
        customdata = np.stack(
            [
                plot_df["JobID"].astype(str).values,                   # [0]
                plot_df["billing"].values.astype(float),               # [1]
                plot_df["Start"].dt.strftime(datetime_fmt).astype(object).values,  # [2]
                plot_df["End"].dt.strftime(datetime_fmt).astype(object).values,    # [3]
                plot_df["duration_min"].values.astype(float),          # [4]
            ],
            axis=-1,
        )
        hovertemplate = (
            "<b>JobID</b>: %{customdata[0]}<br>"
            "<b>Billing</b>: %{customdata[1]:,.0f}<br>"
            "<b>Start</b>: %{customdata[2]}<br>"
            "<b>End</b>: %{customdata[3]}<br>"
            "<b>Durée</b>: %{customdata[4]:.1f} min"
            "<extra></extra>"
        )

    # --- Graphique
    fig = px.bar(
        plot_df,
        x="Start",
        y="duration_min",
        color="billing",
        color_continuous_scale=color_scale,
        labels={
            "Start": "Date de début",
            "duration_min": "Durée (minutes)",
            "billing": "Billing",
        },
        title=(title if not aggregated else f"{title} — agrégé par {aggregate}"),
    )

    # Injecter customdata + hover + style
    fig.update_traces(
        customdata=customdata,
        hovertemplate=hovertemplate,
        marker_line_width=0,  # enlever le liseré qui amincit visuellement
        opacity=0.95,
    )

    # Mise en page
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Durée (minutes)"),
        coloraxis_colorbar=dict(title="billing"),
        margin=dict(l=60, r=20, t=60, b=50),
        bargap=0.05,
        bargroupgap=0.05,
    )

    # Échelle de couleur bornée (optionnel)
    if (color_cmin is not None) or (color_cmax is not None):
        fig.update_coloraxes(cmin=color_cmin, cmax=color_cmax)

    # --- Gestion de la largeur des barres ---
    if bar_width_mode == "category":
        # Barres uniformes, axe catégoriel (espacements non proportionnels au temps)
        fig.update_layout(xaxis_type="category")

    elif bar_width_mode == "fixed_ms":
        # Largeur fixe en ms (par défaut ~0.8 jour)
        if fixed_bar_width is None:
            fixed_bar_width = int(pd.Timedelta(hours=19.2) / pd.Timedelta(milliseconds=1))
        fig.update_traces(width=fixed_bar_width)

    elif bar_width_mode == "auto":
        # Largeur calculée à partir de l'écart médian entre deux Start successifs
        if len(plot_df) >= 2:
            diffs = plot_df["Start"].sort_values().diff().dropna()
            if len(diffs):
                bar_width_ms = (diffs.median() * 0.8) / pd.Timedelta(milliseconds=1)
            else:
                bar_width_ms = (pd.Timedelta(hours=1) / pd.Timedelta(milliseconds=1))
        else:
            bar_width_ms = (pd.Timedelta(hours=1) / pd.Timedelta(milliseconds=1))
        fig.update_traces(width=bar_width_ms)

    # "none" : ne rien faire (laisse la largeur par défaut de Plotly)

    if show:
        fig.show()

    return fig




def plot_completed_jobs_max_gpu_memory(jobs_df, gpu_df):
    """
    Graphique : pour chaque JobID COMPLETED (et gpu_type non vide),
    le maximum de mémoire GPU utilisée (memory_util) extrait de gpu_df.

    Ajouts :
    - Utilise la colonne 'Elapsed' (secondes) du jobs_df, convertie en minutes,
      et l'affiche dans le hover.

    Inputs
    ------
    jobs_df : pd.DataFrame
        Doit contenir : JobID, State, billing, gpu_type, Elapsed (en secondes)
    gpu_df : pd.DataFrame
        Doit contenir : slurmjobid, memory_util

    Output
    ------
    fig : plotly.graph_objects.Figure
    """

    # --- Copies & standardisation des noms de colonnes
    jobs = jobs_df.copy()
    gpu  = gpu_df.copy()
    jobs.columns = [c.strip() for c in jobs.columns]
    gpu.columns  = [c.strip() for c in gpu.columns]

    required_jobs = {"JobID", "State", "billing", "gpu_type", "Elapsed"}
    required_gpu  = {"slurmjobid", "memory_util"}
    if not required_jobs.issubset(jobs.columns):
        raise ValueError(f"jobs_df doit contenir : {required_jobs}")
    if not required_gpu.issubset(gpu.columns):
        raise ValueError(f"gpu_df doit contenir : {required_gpu}")

    # --- Types numériques & nettoyage minimal
    jobs["billing"] = pd.to_numeric(jobs["billing"], errors="coerce")
    jobs["JobID"]   = pd.to_numeric(jobs["JobID"], errors="coerce")
    jobs["Elapsed"] = pd.to_numeric(jobs["Elapsed"], errors="coerce")  # secondes

    gpu["slurmjobid"]  = pd.to_numeric(gpu["slurmjobid"], errors="coerce")
    gpu["memory_util"] = pd.to_numeric(gpu["memory_util"], errors="coerce")

    # --- Filtre : COMPLETED + gpu_type non vide
    completed = (
        jobs[
            (jobs["State"] == "COMPLETED") &
            (jobs["gpu_type"].notna()) & (jobs["gpu_type"].astype(str).str.len() > 0)
        ][["JobID", "billing", "gpu_type", "Elapsed"]]
        .dropna(subset=["JobID"])
    )
    completed["JobID"] = completed["JobID"].astype(int)

    # Convertir Elapsed (sec) -> minutes
    completed["elapsed_min"] = completed["Elapsed"] / 60.0

    # --- Max(memory_util) par slurmjobid
    gpu_max = (
        gpu.dropna(subset=["slurmjobid"])
           .groupby("slurmjobid", as_index=False)["memory_util"]
           .max()
           .rename(columns={
               "slurmjobid": "JobID",
               "memory_util": "max_memory_gb"
           })
    )
    gpu_max["JobID"] = gpu_max["JobID"].astype(int)

    # --- Jointure & tri
    plot_df = (completed
               .merge(gpu_max, on="JobID", how="left")
               .dropna(subset=["max_memory_gb"])
               .sort_values("JobID")
               .reset_index(drop=True))

    # --- Échelle couleur cohérente (sur tous les jobs)
    billing_min = jobs["billing"].min(skipna=True)
    billing_max = jobs["billing"].max(skipna=True)

    # --- Graphique
    fig = px.bar(
        plot_df,
        x="JobID",
        y="max_memory_gb",
        color="billing",
        color_continuous_scale="Viridis",
        labels={
            "JobID": "JobID",
            "max_memory_gb": "Maximum GPU Memory Used (GB)",
            "billing": "Billing"
        },
        title="Maximum GPU Memory Used (GB) — Jobs COMPLETED",
    )

    # Hover enrichi : JobID, mémoire max, billing, gpu_type, elapsed (min)
    # On passe gpu_type et elapsed_min via customdata
    fig.update_traces(
        marker_line_width=0,
        opacity=0.95,
        customdata=plot_df[["gpu_type", "elapsed_min"]].values,
        hovertemplate=(
            "<b>JobID</b>: %{x}<br>"
            "<b>Max GPU memory</b>: %{y:.2f} GB<br>"
            "<b>Billing</b>: %{marker.color:,.0f}<br>"
            "<b>GPU Type</b>: %{customdata[0]}<br>"
            "<b>Elapsed</b>: %{customdata[1]:.1f} min"
            "<extra></extra>"
        )
    )

    # Même gradué pour le billing
    if pd.notna(billing_min) and pd.notna(billing_max):
        fig.update_coloraxes(cmin=float(billing_min), cmax=float(billing_max))

    # Forcer Y max à 40 GB
    fig.update_yaxes(range=[0, 40])

    # Style et barres lisibles
    fig.update_layout(
        xaxis_type="category",
        bargap=0.05,
        bargroupgap=0.02,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50)
    )

    return fig




PALETTE_0_13 = {
    0:  "#1f77b4", 1:  "#ff7f0e", 2:  "#2ca02c", 3:  "#d62728",
    4:  "#9467bd",
    5:  "#2F4B7C",  # code 5 (Inactive)
    6:  "#e377c2", 7:  "#7f7f7f", 8:  "#bcbd22",
    9:  "#17becf", 10: "#1a55FF", 11: "#FF1493",
    12: "#00CED1", 13: "#AAAA11",
}

CODE_DESCRIPTIONS = {
    0:  "Correcte",
    1:  "Moins de 10 minutes",
    2:  "Interactive > 6 heures",
    3:  "Interactive non active pour 30 minutes",
    4:  "Non active > 30 minutes",
    5:  "Inactive",
    6:  "A100 => MIG 1g.5bg",
    7:  "A100 => MIG 2g.10bg",
    8:  "A100 => MIG 3g.20bg",
    9:  "A100 trop de mémoire CPU",
    10: "MIG trop de mémoire CPU",
    11: "Tâche MIG qui demande > 1 MIG",
    12: "MIG 3g.20bg => MIG 2g.10bg",
    13: "MIG 2g.10bg => MIG 1g.5bg",
}

def _parse_code_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, set, np.ndarray)):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums] if nums else []

def plot_job_duration_by_codes_pages(
    df: pd.DataFrame,
    *,
    job_col: str = "JobID",
    duration_col_seconds: str = "Elapsed",
    code_col: str = "code",
    palette: dict = PALETTE_0_13,
    code_descriptions: dict = CODE_DESCRIPTIONS,
    title_base: str = "Durée (minutes) par JobID — segments colorés par code",
    page_size: int = 40,
    sort_by: str = "job",
    ascending: bool = True,
    y_max: Optional[float] = None,
    rotate_xticks: int = 45,
    per_fig_height: int = 340,
    legend_orientation: str = "h",
    save_html_prefix: Optional[str] = None,
    legend_y: float = -0.28,      # ↓↓↓  nouveaux paramètres pour espacement
    bottom_margin: int = 140      # marge basse suffisante pour la légende
) -> List["px.Figure"]:

    data = df.copy()
    for col in [job_col, duration_col_seconds, code_col]:
        if col not in data.columns:
            raise ValueError(f"Colonne manquante: '{col}'")

    data[job_col] = data[job_col].astype(str)
    data[duration_col_seconds] = pd.to_numeric(data[duration_col_seconds], errors="coerce")
    data = data.dropna(subset=[job_col, duration_col_seconds])

    # secondes -> minutes
    data["__duration_min__"] = data[duration_col_seconds] / 60.0

    # codes -> liste d'entiers
    data["__codes_list__"] = data[code_col].apply(_parse_code_list)
    data = data[data["__codes_list__"].map(len) > 0].copy()
    if data.empty:
        raise ValueError("Aucune ligne avec un 'code' non vide.")

    # ordre des JobID
    if sort_by == "duration":
        order_df = (data.groupby(job_col, as_index=False)["__duration_min__"].sum()
                        .sort_values("__duration_min__", ascending=ascending))
        job_order = order_df[job_col].tolist()
        data = data.set_index(job_col).loc[job_order].reset_index()
    else:
        job_order = sorted(data[job_col].unique(), key=lambda s: (len(s), s)) if ascending \
                    else sorted(data[job_col].unique(), key=lambda s: (len(s), s), reverse=True)

    n_jobs = len(job_order)
    n_pages = (n_jobs + page_size - 1) // page_size

    # Ordre global pour la légende
    all_codes_sorted = sorted({c for codes in data["__codes_list__"] for c in codes})
    def _label_for(k: int) -> str:
        return f"{k} — {code_descriptions.get(k, 'Inconnu')}"
    legend_order = [_label_for(k) for k in all_codes_sorted]
    fallback = "#999999"
    color_map_global = { _label_for(k): palette.get(k, fallback) for k in all_codes_sorted }

    figs = []
    for p in range(n_pages):
        start_idx = p * page_size
        end_idx   = min((p + 1) * page_size, n_jobs)
        jobs_slice = job_order[start_idx:end_idx]

        d = data[data[job_col].isin(jobs_slice)].copy()
        d["__n_codes__"]     = d["__codes_list__"].map(len)
        d["__segment_val__"] = d["__duration_min__"] / d["__n_codes__"]

        long_df = d.loc[:, [job_col, "__duration_min__", "__segment_val__", "__codes_list__"]].explode("__codes_list__")
        long_df = long_df.rename(columns={"__codes_list__": "code_segment"})
        long_df["code_segment"] = long_df["code_segment"].astype(int)
        long_df["code_label"]   = long_df["code_segment"].apply(_label_for)
        long_df[job_col] = pd.Categorical(long_df[job_col], categories=jobs_slice, ordered=True)

        present_labels = [lbl for lbl in legend_order if lbl in long_df["code_label"].unique()]
        color_map = {lbl: color_map_global[lbl] for lbl in present_labels}

        fig = px.bar(
            long_df,
            x=job_col,
            y="__segment_val__",
            color="code_label",
            color_discrete_map=color_map,
            category_orders={job_col: jobs_slice, "code_label": present_labels},
            custom_data=["__duration_min__", "code_segment", "code_label"],
            labels={job_col: "JobID", "__segment_val__": "Durée (minutes)", "code_label": "Code"},
            title=f"{title_base} — Groupe {p+1} (jobs {start_idx+1}–{end_idx})",
        )

        fig.update_layout(barmode="stack")
        fig.update_traces(
            marker_line_width=0,
            opacity=0.95,
            hovertemplate=(
                "<b>JobID</b>: %{x}<br>"
                "<b>Durée totale</b>: %{customdata[0]:.1f} min<br>"
                "<b>Part (segment)</b>: %{y:.1f} min<br>"
                "<b>Code</b>: %{customdata[1]}<br>"
                "<b>Définition</b>: %{customdata[2]}<extra></extra>"
            ),
        )

        # Légende sous le graphe + marges suffisantes
        if legend_orientation == "h":
            legend_cfg = dict(orientation="h", yanchor="top", y=legend_y, xanchor="center", x=0.5, traceorder="normal")
        else:
            legend_cfg = dict(orientation="v")

        fig.update_layout(
            template="plotly_white",
            height=max(per_fig_height, 320),
            margin=dict(l=60, r=20, t=70, b=bottom_margin),
            legend=legend_cfg,
            legend_title_text="Code — Définition",
        )

        # Axes X : angle, label, auto‑marge
        fig.update_xaxes(tickangle=rotate_xticks, title_text="JobID", automargin=True)

        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])

        if save_html_prefix:
            idx_str = str(p+1).zfill(len(str(n_pages)))
            fig.write_html(f"{save_html_prefix}{idx_str}.html", include_plotlyjs="cdn")

        figs.append(fig)

    return figs