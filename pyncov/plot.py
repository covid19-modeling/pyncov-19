# -*- coding: utf-8 -*-
from warnings import warn
import numpy as np

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    warn(
        """The pyncov.plot package requires libraries to be installed.
    You can install them with pip install pyncov[all]
    """
    )
    raise ImportError(
        "pyncov.plot requires extra libs to be installed"
    ) from None


def plot_state(sims, state, diff=False, ax=None, index=None, title=None, figsize=(6, 4), alpha=0.05):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    df_a = pd.DataFrame(sims[:, :, state]).transpose()
    df_b = pd.Series(np.mean(sims[:, :, state], axis=0))
    if index is not None:
        df_a = df_a.head(len(index)).set_index(index)
        df_b = pd.Series(np.mean(sims[:, :len(index), state], axis=0), index=index)
    if diff:
        df_a = df_a.diff()
        df_b = df_b.diff()
    df_a.plot.line(ax=ax, alpha=alpha, title=title, color='k', legend=False)
    df_b.plot.line(ax=ax, color='r', linestyle='--')
