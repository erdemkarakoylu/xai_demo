"""Scripts to plot ML model inspection plots."""
import matplotlib.pyplot as pp
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def color_normalization(data, center_at=0):
    vcenter = 0
    vmin, vmax = data.min(), data.max()
    normalize = TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)
    return normalize

def compute_permutation_importances(estimator, X, y, random_state=42, n_repeats=10):
    perm = permutation_importance(
        estimator=estimator, X=X, y=y, raandom_state=random_state, n_repeats=n_repeats)
    return perm

def organize_data(perm_object, n_features, feature_names):
    mean_imps = perm_object.importances_mean
    abs_sort_idx = np.flip(np.abs(mean_imps).argsort())[:n_features]
    mean_imp_ser = pd.Series(
        mean_imps[abs_sort_idx], index=feature_names[abs_sort_idx])
    importances_array = perm_object.importances[np.flip(abs_sort_idx)].T
    return mean_imp_ser, importances_array

def plot_prem_import(X, y, model, feature_names, n_features=15, perm_object=None):
    if perm_object is None:
        perm_object = compute_permutation_importances(model, X, y)
    
    mean_imp_ser, imp_array = organize_data(perm_obj)
    normalize = color_normalization(mean_imp_ser)
    
    f, axs = pp.subplots(nrows=1, ncols=2, figsize=(6, 10))
    sb.barplot(
    x=mean_imp_ser.values, y=mean_imp_ser.index.to_list(), orient='h',
    hue=mean_imp_ser.values, hue_norm=normalize, palette='PRGn', legend=False, edgecolor='k',
    ax=axs[0],
    )
    axs[0].axvline(x=0, color='r', lw=2, ls='--')
    axs[1].boxplot(X, vert=False);
    axs[1].set_yticklabels([])
    axs[1].axvline(x=0, color='r', lw=2, ls='--')
    f.tight_layout()
