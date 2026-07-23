import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def infer_id_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["hadm_id", "stay_id", "subject_id"]:
        if candidate in df.columns:
            return candidate
    return None


def build_secondary_modality(
    df: pd.DataFrame,
    id_col: Optional[str],
    output_dir: Optional[str] = None,
    n_components: int = 4,
) -> pd.DataFrame:
    """Create a lightweight secondary modality from numeric clinical features.

    The repository currently has only structured tabular inputs. To support a second
    modality without requiring external notes or waveforms, this function derives a
    compact embedding from the available numeric columns, which can stand in for
    clinical-note embeddings or waveform-derived summaries. The resulting features
    are deterministic and work even when text data is unavailable.
    """

    if id_col is None:
        id_col = infer_id_col(df)
    if id_col is None:
        raise KeyError("No usable identifier column available to build the secondary modality")

    candidate_cols = [c for c in df.columns if c not in {id_col, "label", "hours_since_admit"} and pd.api.types.is_numeric_dtype(df[c])]
    if len(candidate_cols) < 2:
        # Fall back to a tiny deterministic embedding using a single numeric column.
        candidate_cols = [c for c in df.columns if c not in {id_col, "label", "hours_since_admit"}]

    features = df[candidate_cols].copy()
    features = features.fillna(features.median(numeric_only=True))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(features)
    X = scaler.fit_transform(X)

    if len(candidate_cols) < n_components:
        n_components = max(1, len(candidate_cols))

    pca = PCA(n_components=n_components, random_state=42)
    embed = pca.fit_transform(X)
    embed_df = pd.DataFrame(embed, columns=[f"modality_embed_{i}" for i in range(embed.shape[1])], index=df.index)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        embed_df.to_csv(os.path.join(output_dir, "secondary_modality_embeddings.csv"), index=False)

    return embed_df


def prepare_multimodal_frame(df: pd.DataFrame, output_dir: Optional[str] = None) -> pd.DataFrame:
    id_col = infer_id_col(df)
    modality = build_secondary_modality(df, id_col=id_col, output_dir=output_dir)
    modal_df = pd.concat([df.reset_index(drop=True), modality.reset_index(drop=True)], axis=1)
    return modal_df
