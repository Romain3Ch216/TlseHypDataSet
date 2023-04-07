import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'vis_proportions'
]


def vis_proportions(dataset, proportions):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ticks = range(1, len(proportions[0])+1)
    x_ticks = np.arange(1, dataset.n_classes+1, 5)

    ax[0, 0].bar(ticks, proportions[0], color=dataset.colors)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_xticks(x_ticks)

    ax[0, 1].bar(ticks, proportions[1], color=dataset.colors)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_xticks(x_ticks)

    ax[1, 0].bar(ticks, proportions[2], color=dataset.colors)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_xticks(x_ticks)

    ax[1, 1].bar(ticks, proportions[3], color=dataset.colors)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_xticks(x_ticks)

    return fig



