import matplotlib.pyplot as plt
import seaborn as sns   

# Set global plot style
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
    """ Create scatter plots for specified variable groups. """
    n_cols = 2
    n_rows = (len(var_groups) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
    axes = axes.flatten()

    # create plots
    for i, vars in enumerate(var_groups):
        sns.scatterplot(data=df, x=vars[0], y=vars[1], hue='form', ax=axes[i], legend=(i==0))
        axes[i].set_title(f"{vars[0]} vs {vars[1]}")

    # Remove empty subplots
    for i in range(len(var_groups), len(axes)):
        fig.delaxes(axes[i])

    # Get legend from the first plot
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend():
        axes[0].get_legend().remove()

    # add legend to the figure
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left')

    plt.tight_layout()
    plt.show()


def plot_explorations(matrices, similarities_explorations, ablation_explorations, weights_explorations):
    """ Plot heatmaps for each exploration's similarity matrices. """
    for main_key, sub_matrices in matrices.items():
        n_plots = len(sub_matrices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle(main_key, fontsize=16)
        
        for i, (sub_key, matrix) in enumerate(sub_matrices.items()):
            sns.heatmap(matrix, ax=axes[i], cmap='viridis', square=True, cbar_kws={'shrink': 0.8})
            axes[i].tick_params(axis='both', which='major', labelsize=10)
            # Get the parameter values for this exploration
            params = next((params for name, params in [*similarities_explorations.items(), 
                          *ablation_explorations.items(), *weights_explorations.items()] 
                          if name == sub_key), {})
            title = f"{sub_key}\n{params}"
            axes[i].set_title(title, fontsize=10)
        
        plt.tight_layout()
        plt.show()