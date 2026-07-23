import os
import tempfile
import unittest

import pandas as pd

from src.multimodal_support import build_secondary_modality, infer_id_col


class MultimodalSupportTests(unittest.TestCase):
    def test_infer_id_col_prefers_common_identifiers(self) -> None:
        df = pd.DataFrame({"subject_id": [1, 2], "label": [0, 1]})
        self.assertEqual(infer_id_col(df), "subject_id")

    def test_build_secondary_modality_creates_embedding_columns(self) -> None:
        df = pd.DataFrame({
            "hadm_id": [101, 102, 103],
            "label": [0, 1, 0],
            "age": [60.0, 75.0, 55.0],
            "hr_max": [110.0, 130.0, 100.0],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            modality = build_secondary_modality(df, "hadm_id", tmp_dir, n_components=3)
            self.assertEqual(modality.shape[0], df.shape[0])
            self.assertEqual(modality.shape[1], 3)
            self.assertIn("modality_embed_0", modality.columns)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "secondary_modality_embeddings.csv")))


if __name__ == "__main__":
    unittest.main()
