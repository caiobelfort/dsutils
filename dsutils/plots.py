import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(dataframe, var, target, **kwargs):
    """
    Plot a distribution of a variable over a target categorical one

    Parameters
    ----------
    dataframe: pandas.DataFrame
    var: str
    target: str
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


def plot_correlation_map(df):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )
