from pandas import plotting


# Code:https://stackoverflow.com/questions/11640243/pandas-plot-multiple-y-axes
def plot_multi(data, cols=None, spacing=.1, vline=None, **kwargs):
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return
    colors = getattr(getattr(plotting, '_matplotlib').style,
                     '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0], fontsize=16)
    ax.tick_params(axis='y', colors=colors[0])
    lines, labels = ax.get_legend_handles_labels()
    if vline is not None:
        for v in vline:
            ax.axvline(v, color='r', linestyle='--', lw=2)

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax.set_ylabel(ylabel=cols[n])
        ax.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(
            ax=ax, label=cols[n], color=colors[n % len(colors)])
        ax.tick_params(axis='y', colors='black')

        # Proper legend position
        lines, labels = ax.get_legend_handles_labels()

    ax.legend(lines, labels, loc=0, fontsize=16)

    return ax


def plot_multi_axes(data, cols=None, spacing=.1, vline=None, **kwargs):
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return
    colors = getattr(getattr(plotting, '_matplotlib').style,
                     '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0], fontsize=16)
    ax.tick_params(axis='y', colors=colors[0])
    lines, labels = ax.get_legend_handles_labels()
    if vline is not None:
        for v in vline:
            ax.axvline(v, color='r', linestyle='--', lw=2)

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n],
                                  color=colors[n % len(colors)])
        ax_new.set_ylabel(ylabel=cols[n], fontsize=16)
        ax_new.tick_params(axis='y', colors=colors[n % len(colors)])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0, fontsize=16)

    return ax
