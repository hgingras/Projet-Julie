import pandas as pd
from ast import literal_eval
from typing import Mapping, Iterable, Union, Optional, Any
from collections import defaultdict

def _to_list_safe(x: Any) -> list[int]:
    """Convertit proprement une cellule vers une liste d'entiers (préserve scalaires)."""
    # Déjà liste
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    # NaN
    if pd.isna(x):
        return []
    # Chaîne potentiellement de type "['1', 2]" etc.
    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                val = literal_eval(s)
                if isinstance(val, list):
                    return [int(v) for v in val if pd.notna(v)]
            except Exception:
                pass
        return []
    # Scalaire numérique -> on garde
    try:
        return [int(x)]
    except Exception:
        return []

def _append_unique(lst: list[int], c: int) -> list[int]:
    return lst if c in lst else lst + [c]

def add_code_for_jobs(
    df: pd.DataFrame,
    jobids_or_map: Union[Iterable[int], Mapping[int, int]],
    code: Optional[int] = None,
    *,
    jobid_col: str = "JobID",
    codes_col: str = "code",   # <-- on vise la colonne existante 'code'
) -> pd.DataFrame:
    # Garde-fous
    if jobid_col not in df.columns:
        raise KeyError(f"Colonne '{jobid_col}' introuvable. Colonnes: {list(df.columns)}")

    # Normaliser la colonne identifiant (utile si elle est lue en 'object')
    df[jobid_col] = pd.to_numeric(df[jobid_col], errors="coerce").astype("Int64")

    # Préparer la colonne cible en liste
    if codes_col not in df.columns:
        df[codes_col] = [[] for _ in range(len(df))]
    else:
        df[codes_col] = df[codes_col].apply(_to_list_safe)

    # Deux modes: mapping {job: code} OU iterable de jobids (+ code)
    if isinstance(jobids_or_map, Mapping):
        by_code: dict[int, list[int]] = defaultdict(list)
        for j, c in jobids_or_map.items():
            try:
                by_code[int(c)].append(int(j))
            except Exception:
                pass
        for c_int, jobs in by_code.items():
            if not jobs:
                continue
            jobs = [int(x) for x in jobs]
            mask = df[jobid_col].isin(jobs)
            if not mask.any():
                continue
            df.loc[mask, codes_col] = df.loc[mask, codes_col].apply(
                lambda lst, cc=c_int: _append_unique(lst, cc)
            )
    else:
        if code is None:
            raise ValueError("Lorsque `jobids_or_map` est un Iterable, `code` est requis.")
        job_list: list[int] = []
        for j in jobids_or_map:
            try:
                job_list.append(int(j))
            except Exception:
                pass
        if not job_list:
            return df
        mask = df[jobid_col].isin(job_list)
        if not mask.any():
            return df
        df.loc[mask, codes_col] = df.loc[mask, codes_col].apply(
            lambda lst, cc=int(code): _append_unique(lst, cc)
        )

    return df