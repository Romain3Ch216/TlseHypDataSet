import pdb

import matplotlib.pyplot as plt
import numpy as np
from TlseHypDataSet.utils.spectral import full_spectra


__all__ = [
    'vis_proportions',
    'vis_selection'
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


def vis_selection(dataset, selection, figures='multi', tick_size=15, label_fontsize=15, title_size=18):
    n_classes = dataset.n_classes
    class_ids = np.arange(1, n_classes + 1)
    if figures == 'multi':
        figures = []
        for class_id in class_ids:
            if class_id in selection:
                indices = selection[class_id]
                fig, ax = plt.subplots()
                for j in range(len(indices)):
                    sp, _ = dataset.__getitem__(indices[j])
                    sp = full_spectra(sp.reshape(1, -1), dataset.bbl).reshape(-1)
                    ax.plot(dataset.wv, sp, alpha=0.5)
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title('{} - {}'.format(class_id, dataset.labels[class_id]), fontsize=title_size)
                ax.tick_params(axis='x', labelsize=tick_size)
                ax.tick_params(axis='y', labelsize=tick_size)
                ax.set_xlabel('Wavelength (µm)', fontsize=label_fontsize)
                ax.set_ylabel('Reflectance', fontsize=label_fontsize)
                figures.append(fig)
        return figures
    else:
        n_cols = 2
        n_rows = max(2, n_classes // n_cols + n_classes % n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 8 * n_rows),
                               gridspec_kw={'height_ratios': [1 for i in range(n_rows)]})
        for i, class_id in enumerate(class_ids):
            if class_id in selection:
                indices = selection[class_id]
                for j in range(len(indices)):
                    sp, _ = dataset.__getitem__(indices[j])
                    sp = full_spectra(sp.reshape(1, -1), dataset.bbl).reshape(-1)
                    ax[i // 2, i % 2].plot(dataset.wv, sp, alpha=0.5)
                ax[i // 2, i % 2].set_ylim(0, 1)
                ax[i // 2, i % 2].grid(True, linestyle='--', alpha=0.5)
            ax[i // 2, i % 2].set_title('{} - {}'.format(class_id, dataset.labels[class_id]))
            ax[i // 2, i % 2].set_xlabel('Wavelength (µm)')
            ax[i // 2, i % 2].set_ylabel('Reflectance')

        return fig
