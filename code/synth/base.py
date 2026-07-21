"""Shared preprocessing helpers for the torch-based synthesizers
(TVAE-lite, CTGAN-lite). Handles numeric scaling and categorical one-hot
encoding with a fixed column/category schema learned at fit time.
"""
import numpy as np
import pandas as pd


class TabularTransformer:
    """Fits a numeric scaler + categorical one-hot schema on a training
    frame, and can transform/inverse-transform frames to/from a single
    numeric matrix suitable for a neural net.

    numeric_mode: 'standard' (zero-mean unit-var) or 'minmax' (scaled to
    [-1, 1], used by the GAN so generator tanh outputs are meaningful).
    """

    def __init__(self, numeric_cols, categorical_cols, numeric_mode="standard"):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_mode = numeric_mode
        self.num_stats = {}  # col -> (loc, scale) ; for minmax loc=min, scale=(max-min)/2 (so center=(min+max)/2 handled separately)
        self.cat_categories = {}  # col -> list of categories (sorted by frequency desc)
        self.cat_freq = {}  # col -> np.array of empirical frequencies (fallback sampling)

    def fit(self, df):
        for c in self.numeric_cols:
            vals = df[c].to_numpy(dtype=float)
            if self.numeric_mode == "standard":
                loc = float(np.mean(vals))
                scale = float(np.std(vals))
                if scale < 1e-8:
                    scale = 1.0
            else:  # minmax -> [-1, 1]
                lo, hi = float(np.min(vals)), float(np.max(vals))
                if hi - lo < 1e-8:
                    hi = lo + 1.0
                loc = (lo + hi) / 2.0
                scale = (hi - lo) / 2.0
            self.num_stats[c] = (loc, scale)
        for c in self.categorical_cols:
            vc = df[c].astype(str).value_counts()
            self.cat_categories[c] = vc.index.tolist()
            self.cat_freq[c] = (vc.to_numpy(dtype=float) / vc.sum())
        return self

    @property
    def cat_dims(self):
        return [len(self.cat_categories[c]) for c in self.categorical_cols]

    @property
    def output_dim(self):
        return len(self.numeric_cols) + sum(self.cat_dims)

    def transform(self, df):
        """Returns a float32 numpy array [n, output_dim]: numeric block
        first (scaled), then one-hot categorical blocks concatenated in
        categorical_cols order."""
        n = len(df)
        blocks = []
        for c in self.numeric_cols:
            loc, scale = self.num_stats[c]
            blocks.append(((df[c].to_numpy(dtype=float) - loc) / scale).reshape(-1, 1))
        for c in self.categorical_cols:
            cats = self.cat_categories[c]
            idx = {cat: i for i, cat in enumerate(cats)}
            col_vals = df[c].astype(str).to_numpy()
            oh = np.zeros((n, len(cats)), dtype=np.float32)
            for row, v in enumerate(col_vals):
                j = idx.get(v)
                if j is not None:
                    oh[row, j] = 1.0
                # unseen category (shouldn't happen within the same fitted
                # pool, but guard anyway): leave as all-zero row
            blocks.append(oh)
        if not blocks:
            return np.zeros((n, 0), dtype=np.float32)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def inverse_transform_numeric(self, num_block):
        """num_block: [n, n_numeric] scaled values -> dict col -> array."""
        out = {}
        for i, c in enumerate(self.numeric_cols):
            loc, scale = self.num_stats[c]
            out[c] = num_block[:, i] * scale + loc
        return out

    def inverse_transform_categorical_argmax(self, logits_list):
        """logits_list: list aligned with categorical_cols, each [n, n_cats]
        (raw scores, softmax not required for argmax). Returns dict
        col -> array of category strings."""
        out = {}
        for c, logits in zip(self.categorical_cols, logits_list):
            cats = np.array(self.cat_categories[c])
            idx = np.argmax(logits, axis=1)
            out[c] = cats[idx]
        return out

    def assemble_df(self, numeric_dict, categorical_dict, n):
        data = {}
        for c in self.numeric_cols:
            data[c] = numeric_dict[c]
        for c in self.categorical_cols:
            data[c] = categorical_dict[c]
        return pd.DataFrame(data)

    def cat_slices(self):
        """Returns list of (start, end) index slices into the one-hot part
        of the transformed vector, aligned with categorical_cols order."""
        slices = []
        start = len(self.numeric_cols)
        for dim in self.cat_dims:
            slices.append((start, start + dim))
            start += dim
        return slices
