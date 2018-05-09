import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import typing


def plot_distribution(dataframe, var, target=None, **kwargs):
    """
    Plot a distribution of a variable over a target categorical one

    Args
    ----------
    dataframe: pandas.DataFrame
    var: str
    target: str or None
        A categorical variable
    kwargs: additional arguments
    """

    row = kwargs.get('row', None)
    col = kwargs.get('col', None)

    size = kwargs.get('size', None)
    aspect = kwargs.get('aspect', None)

    facet = sns.FacetGrid(
        dataframe,
        hue=target,
        size=size if size else 3,
        aspect=aspect if aspect else 4,
        row=row,
        col=col
    )

    facet.map(sns.kdeplot, var, shade=True)
    facet.set(title="Distribution of '{}' over '{}'".format(var, target))

    # Defines limits
    if kwargs.get('clean', False):
        xmin = dataframe[var].quantile(0.02)
        xmax = dataframe[var].quantile(0.98)
        facet.set(xlim=(xmin, xmax))

    facet.add_legend()
    plt.show()


def plot_correlation_map(dataframe, method='pearson', **kwargs):
    """
    Plot a correlation map of a table

    Parameters
    ----------
    dataframe: pandas.DataFrame
    method: {'pearson', 'kendal', 'spearman'}
    kwargs: additional arguments

    """

    figsize = kwargs.get('figsize', None)
    cmap = kwargs.get('cmap', None)

    correlation = dataframe.corr(method=method)

    if figsize is None:
        figsize = (12, 10)
    if cmap is None:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlation,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )
    plt.show()


def plot_silhouette_score(
        samples: np.ndarray or pd.DataDrame,
        labels: np.ndarray or pd.Series,
        silhouette: np.ndarray or pd.Series = None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        cmap: str = None,
        figsize: typing.Tuple[int, int] = None):
    """

    Plot the silhouette score of samples based on pre-computed labels

    Args:
        samples: The feature matrix, can be a numpy 2D array or a pandas DataFrame
        labels: The pre-computed labels.
                If samples is a pandas DataFrame, labels must be a pandas Series with same index.
        silhouette: The silhoute values of each sample.
                    If None, they are computed, if not, the object must be a pandas Series with same index of samples
        title: title name
        xlabel: X axis label
        ylabel: Y axis label
        cmap: A colormap to draw clusters silhouettes
        figsize: Figure size in inches

    Returns:
        A figure and axis object
    """
    from sklearn.metrics import silhouette_samples

    # Compute silhouette if is None
    if silhouette is None:
        silhouette = silhouette_samples(samples, labels)
        if isinstance(samples, pd.DataFrame):
            silhouette = pd.Series(silhouette, index=samples.index)

    # Configure plot attributes
    figsize = figsize if figsize is not None else (16, 10)
    title = title if title is not None else 'Silhouette Score'
    xlabel = 'silhoette coefficients' if xlabel is None else xlabel
    ylabel = 'Samples' if ylabel is None else ylabel

    cmap = cm.get_cmap(cmap) if cmap is not None else cm.get_cmap('nipy_spectral')


    # Begin plot
    n_clusters = len(labels.unique())

    fig, ax = plt.subplots(figsize = figsize)

    ax.set_xlim([-0.1, 1])
    ax.set_ylim(0, len(samples) + (n_clusters + 1) * 10)


    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette = silhouette[labels == i]

        if isinstance(ith_cluster_silhouette, pd.Series):
            ith_cluster_silhouette.sort_values(inplace=True)
        else:
            ith_cluster_silhouette.sort()



        size_cluster_i = ith_cluster_silhouette.shape[0]

        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            label= 'Label %d' % i
        )
        y_lower = y_upper + 10

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.plot()