import matplotlib.pyplot as plt
import seaborn as sns   

plt.rcParams.update({
    'lines.linewidth': 2,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 2
})


def plot_scatter_for_variable_groups(df, var_groups):
    n_cols = 2
    n_rows = (len(var_groups) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
    axes = axes.flatten()

    # Create scatter plots for each variable group
    for i, vars in enumerate(var_groups):
        sns.scatterplot(data=df, x=vars[0], y=vars[1], hue='form', ax=axes[i], legend=(i==0))
        axes[i].set_title(f"{vars[0]} vs {vars[1]}")

    # Remove empty subplots
    for i in range(len(var_groups), len(axes)):
        fig.delaxes(axes[i])

    # Get legend from the first plot (don't replot!)
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend():
        axes[0].get_legend().remove()

    # Add single legend to the entire figure
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left')

    plt.tight_layout()
    plt.show()