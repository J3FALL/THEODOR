import os
from copy import deepcopy
from glob import glob, iglob
from math import ceil, log2
from os import remove
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image
from imageio import get_writer, imread

from fedot.core.chains.chain import Chain, as_nx_graph
from fedot.core.utils import default_fedot_data_dir


class ComposerVisualiser:
    default_data_dir = default_fedot_data_dir()
    temp_path = os.path.join(default_data_dir, 'composing_history')
    if 'composing_history' not in os.listdir(default_data_dir):
        os.mkdir(temp_path)
    gif_prefix = 'for_gif_'

    @staticmethod
    def visualise(chain: Chain, save_path: Optional[str] = None):
        try:
            graph, node_labels = as_nx_graph(chain=chain)
            pos = node_positions(graph.to_undirected())
            plt.figure(figsize=(10, 16))
            nx.draw(graph, pos=pos,
                    with_labels=True, labels=node_labels,
                    font_size=12, font_family='calibri', font_weight='bold',
                    node_size=7000, width=2.0,
                    node_color=colors_by_node_labels(node_labels), cmap='Set3')
            if not save_path:
                plt.show()
            else:
                plt.savefig(save_path)
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    @staticmethod
    def _visualise_chains(chains, fitnesses):
        fitnesses = deepcopy(fitnesses)
        last_best_chain = chains[0]

        prev_fit = fitnesses[0]

        for ch_id, chain in enumerate(chains):
            graph, node_labels = as_nx_graph(chain=chain)
            pos = node_positions(graph.to_undirected())
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.title('Current chain')
            nx.draw(graph, pos=pos,
                    with_labels=True, labels=node_labels,
                    font_size=12, font_family='calibri', font_weight='bold',
                    node_size=scaled_node_size(chain.length), width=2.0,
                    node_color=colors_by_node_labels(node_labels), cmap='Set3')
            path = f'{ComposerVisualiser.temp_path}ch_{ch_id}.png'
            plt.savefig(path, bbox_inches='tight')

            plt.cla()
            plt.clf()
            plt.close('all')

            path_best = f'{ComposerVisualiser.temp_path}best_ch_{ch_id}.png'

            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_chain = chain
            prev_fit = fitnesses[ch_id]

            best_graph, best_node_labels = as_nx_graph(chain=last_best_chain)
            pos = node_positions(best_graph.to_undirected())
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.title(f'Best chain after {round(ch_id)} evals')
            nx.draw(best_graph, pos=pos,
                    with_labels=True, labels=best_node_labels,
                    font_size=12, font_family='calibri', font_weight='bold',
                    node_size=scaled_node_size(chain.length), width=2.0,
                    node_color=colors_by_node_labels(best_node_labels), cmap='Set3')

            plt.savefig(path_best, bbox_inches='tight')

            plt.cla()
            plt.clf()
            plt.close('all')

    @staticmethod
    def _visualise_convergence(fitness_history):
        fitness_history = deepcopy(fitness_history)
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts_set = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {'ts': ts_set, 'fitness': [-f for f in fitness_history]})

        ind = 0
        for ts in ts_set:
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]

            ind = ind + 1
            plt.plot(df['ts'], df['fitness'], label='Composer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)

            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')

            path = f'{ComposerVisualiser.temp_path}{ind}.png'
            plt.savefig(path, bbox_inches='tight')

            plt.cla()
            plt.clf()
            plt.close('all')

    @staticmethod
    def visualise_history(chains, fitnesses):
        print('START VISUALISATION')
        try:
            ComposerVisualiser._clean(with_gif=True)
            ComposerVisualiser._visualise_chains(chains, fitnesses)
            ComposerVisualiser._visualise_convergence(fitnesses)
            ComposerVisualiser._merge_images(len(chains))
            ComposerVisualiser._combine_gifs()
            ComposerVisualiser._clean()
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    @staticmethod
    def _merge_images(num_images):
        for img_idx in (range(1, num_images)):
            images = list(map(Image.open, [f'{ComposerVisualiser.temp_path}ch_{img_idx}.png',
                                           f'{ComposerVisualiser.temp_path}best_ch_{img_idx}.png',
                                           f'{ComposerVisualiser.temp_path}{img_idx}.png']))
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            new_im.save(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}{img_idx}.png')

    @staticmethod
    def _combine_gifs():
        files = [file_name for file_name in
                 iglob(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}*.png')]
        files_idx = [int(file_name[len(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}'):(
                len(file_name) - len('.png'))]) for
                     file_name in
                     iglob(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}*.png')]
        files = [file for _, file in sorted(zip(files_idx, files))]

        with get_writer(f'{ComposerVisualiser.temp_path}final_{str(time())}.gif', mode='I', duration=0.5) as writer:
            for filename in files:
                image = imread(filename)
                writer.append_data(image)

    @staticmethod
    def _clean(with_gif=False):
        try:
            files = glob(f'{ComposerVisualiser.temp_path}*.png')
            if with_gif:
                files += glob(f'{ComposerVisualiser.temp_path}*.gif')
            for file in files:
                remove(file)
        except Exception as ex:
            print(ex)


def colors_by_node_labels(node_labels: dict):
    colors = [color for color in range(len(node_labels.keys()))]
    return colors


def scaled_node_size(nodes_amount):
    size = int(7000.0 / max(ceil(log2(nodes_amount)), 1))
    return size


def node_positions(graph: nx.Graph):
    if not nx.is_tree(graph):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    return nx.drawing.nx_pydot.graphviz_layout(graph, prog='dot')
