import pandas as pd
from ast import literal_eval
from typing import Mapping, Iterable, Union, Optional


def _to_list(x) -> list[int]:
    if isinstance(x, list):
        return [int(v) for v in x if pd.notna(v)]
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(x, str):
        try:
            val = literal_eval(x.strip())
            if isinstance(val, list):
                return [int(v) for v in val if pd.notna(v)]
        except Exception:
            pass
        return []
    try:
        return [int(x)]
    except Exception:
        return []


def add_code_for_jobs(
    df: pd.DataFrame,
    jobids_or_map: Union[Iterable[int], Mapping[int, int]],
    code: Optional[int] = None,
    *,
    jobid_col: str = "JobID",
    codes_col: str = "code",
) -> pd.DataFrame:
    if jobid_col not in df.columns:
        raise KeyError(f"Colonne '{jobid_col}' introuvable. Colonnes: {list(df.columns)}")

    # Unifier les deux modes en un seul mapping {jobid: code}
    if isinstance(jobids_or_map, Mapping):
        job_code_map = {int(j): int(c) for j, c in jobids_or_map.items()}
    else:
        if code is None:
            raise ValueError("Lorsque `jobids_or_map` est un Iterable, `code` est requis.")
        job_code_map = {int(j): int(code) for j in jobids_or_map}

    df[jobid_col] = pd.to_numeric(df[jobid_col], errors="coerce").astype("Int64")

    if codes_col not in df.columns:
        df[codes_col] = [[] for _ in range(len(df))]
    else:
        df[codes_col] = df[codes_col].apply(_to_list)

    # Aligner les codes à ajouter avec les lignes du DataFrame
    new_codes = df[jobid_col].map(job_code_map)  # NaN si jobid absent du mapping
    mask = new_codes.notna()

    updated = pd.Series(
        [lst if c in lst else lst + [c]
         for lst, c in zip(df.loc[mask, codes_col], new_codes[mask].astype(int))],
        index=df.index[mask],
        dtype=object,
    )
    df.loc[mask, codes_col] = updated

    return df