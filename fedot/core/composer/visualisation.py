import itertools
import numpy as np
import seaborn as sns
import os
from copy import deepcopy
from deap import tools
from glob import glob, iglob
from math import ceil, log2
from os import remove
from time import time
from typing import (Any, Optional, Tuple, List)

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
            if type(fitnesses[0]) is not list:
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
    def create_gif_using_images(gif_path: str, files: List[str]):
        with get_writer(gif_path, mode='I', duration=0.5) as writer:
            for filename in files:
                image = imread(filename)
                writer.append_data(image)

    @staticmethod
    def _combine_gifs():
        files = [file_name for file_name in
                 iglob(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}*.png')]
        files_idx = [int(file_name[len(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}'):(
                len(file_name) - len('.png'))]) for
                     file_name in
                     iglob(f'{ComposerVisualiser.temp_path}{ComposerVisualiser.gif_prefix}*.png')]
        files = [file for _, file in sorted(zip(files_idx, files))]

        ComposerVisualiser.create_gif_using_images(gif_path=f'{ComposerVisualiser.temp_path}final_{str(time())}.gif',
                                                   files=files)

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

    @staticmethod
    def objectives_lists(individuals: List[Any], objectives_numbers: Tuple[int] = None):
        num_of_objectives = len(objectives_numbers) if objectives_numbers else len(individuals[0].fitness.values)
        objectives_numbers = objectives_numbers if objectives_numbers else [i for i in range(num_of_objectives)]
        objectives_values_set = [[] for _ in range(num_of_objectives)]
        for obj_num in range(num_of_objectives):
            for individual in individuals:
                value = individual.fitness.values[objectives_numbers[obj_num]]
                objectives_values_set[obj_num].append(value if value > 0 else -value)
        return objectives_values_set

    @staticmethod
    def objectives_transform(individuals: List[List[Any]], objectives_numbers: Tuple[int] = None):
        objectives_numbers = [i for i in range(
            len(individuals[0][0].fitness.values))] if not objectives_numbers else objectives_numbers
        all_inds = list(itertools.chain(*individuals))
        all_objectives = [[ind.fitness.values[i] for ind in all_inds] for i in objectives_numbers]
        all_objectives = list(
            map(lambda obj_values: obj_values if obj_values[0] > 0 else list(np.array(obj_values) * (-1)),
                all_objectives))
        return all_objectives

    @staticmethod
    def create_boxplot(individuals: List[Any], generation_num: int = None,
                       objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'), file_name: str = 'obj_boxplots.png',
                       folder: str = f'../../tmp/boxplots', y_limits: Tuple[float] = None):

        fig, ax = plt.subplots()
        ax.set_title(f'Generation: {generation_num}', fontsize=15)
        objectives = ComposerVisualiser.objectives_lists(individuals)
        df_objectives = pd.DataFrame({objectives_names[i]: objectives[i] for i in range(len(objectives))})
        sns.boxplot(data=df_objectives, palette="Blues")
        if y_limits:
            plt.ylim(y_limits[0], y_limits[1])
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')
        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    @staticmethod
    def boxplots_gif_create(individuals: List[List[Any]], objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'),
                            folder: str = None):
        objectives = ComposerVisualiser.objectives_transform(individuals)
        objectives = list(itertools.chain(*objectives))
        min_y, max_y = min(objectives), max(objectives)
        files = []
        folder = f'{ComposerVisualiser.temp_path}' if folder is None else folder
        for generation_num, individuals_in_genaration in enumerate(individuals):
            file_name = f'{generation_num}.png'
            ComposerVisualiser.create_boxplot(individuals_in_genaration, generation_num, objectives_names,
                                              file_name=file_name, folder=folder, y_limits=(min_y, max_y))
            files.append(f'{folder}/{file_name}')
        ComposerVisualiser.create_gif_using_images(gif_path=f'{folder}/boxplots_history.gif', files=files)
        for file in files:
            remove(file)
        plt.cla()
        plt.clf()
        plt.close('all')

    @staticmethod
    def visualise_pareto(archive: Any, objectives_numbers: Tuple[int] = (0, 1),
                         objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'),
                         file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                         folder: str = f'../../tmp/pareto',
                         generation_num: int = None, individuals: List[Any] = None, minmax_x: List[float] = None,
                         minmax_y: List[float] = None):

        pareto_obj_first, pareto_obj_second = [], []
        for i in range(len(archive.items)):
            fit_first = archive.items[i].fitness.values[objectives_numbers[0]]
            pareto_obj_first.append(fit_first if fit_first > 0 else -fit_first)
            fit_second = archive.items[i].fitness.values[objectives_numbers[1]]
            pareto_obj_second.append(fit_second if fit_second > 0 else -fit_second)

        fig, ax = plt.subplots()

        if individuals is not None:
            obj_first, obj_second = [], []
            for i in range(len(individuals)):
                fit_first = individuals[i].fitness.values[objectives_numbers[0]]
                obj_first.append(fit_first if fit_first > 0 else -fit_first)
                fit_second = individuals[i].fitness.values[objectives_numbers[1]]
                obj_second.append(fit_second if fit_second > 0 else -fit_second)
            ax.scatter(obj_first, obj_second, c='green')

        ax.scatter(pareto_obj_first, pareto_obj_second, c='red')
        plt.plot(pareto_obj_first, pareto_obj_second, color='r')

        if generation_num is not None:
            ax.set_title(f'Pareto front, Generation: {generation_num}', fontsize=15)
        else:
            ax.set_title('Pareto front', fontsize=15)
        plt.xlabel(objectives_names[0], fontsize=15)
        plt.ylabel(objectives_names[1], fontsize=15)
        plt.xlim(minmax_x[0], minmax_x[1])
        plt.ylim(minmax_y[0], minmax_y[1])

        fig.set_figwidth(8)
        fig.set_figheight(8)
        if save:
            if not os.path.isdir('../../tmp'):
                os.mkdir('../../tmp')
            if not os.path.isdir(f'{folder}'):
                os.mkdir(f'{folder}')

            path = f'{folder}/{file_name}'
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close('all')

    @staticmethod
    def pareto_gif_create(pareto_fronts: List[tools.ParetoFront], individuals: List[List[Any]] = None,
                          objectives_numbers: Tuple[int] = (1, 0),
                          objectives_names: Tuple[str] = ('Complexity', 'ROC-AUC')):
        files = []
        array_for_analysis = individuals if individuals else pareto_fronts
        all_objectives = ComposerVisualiser.objectives_transform(array_for_analysis, objectives_numbers)
        min_x, max_x = min(all_objectives[0]) - 0.01, max(all_objectives[0]) + 0.01
        min_y, max_y = min(all_objectives[1]) - 0.01, max(all_objectives[1]) + 0.01
        folder = f'{ComposerVisualiser.temp_path}'
        for i, front in enumerate(pareto_fronts):
            file_name = f'pareto{i}.png'
            ComposerVisualiser.visualise_pareto(front, file_name=file_name, save=True, show=False,
                                                folder=folder, generation_num=i, individuals=individuals[i],
                                                minmax_x=[min_x, max_x], minmax_y=[min_y, max_y],
                                                objectives_numbers=objectives_numbers,
                                                objectives_names=objectives_names)
            files.append(f'{folder}/{file_name}')

        ComposerVisualiser.create_gif_using_images(gif_path=f'{folder}/pareto_history.gif', files=files)
        for file in files:
            remove(file)

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
