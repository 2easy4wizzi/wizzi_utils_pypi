import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from wizzi_utils.pyplot import pyplot_tools as pyplt
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt

SLEEP = 0.2  # sleep between iterations
PLT_MAX_TIME = 2  # close plt after x seconds - minimum 2


def get_colors_formats_test():
    mt.get_function_name(ack=True, tabs=0)
    color_str = 'orange'
    rgb = pyplt.get_rgb_color(color_str)
    rgba = pyplt.get_rgba_color(color_str)
    bgr = pyplt.get_bgr_color(color_str)
    print('\tcolor {}: RGB={}, RGBA={}, BGR={}'.format(color_str, rgb, rgba, bgr))
    return


def RGBA_to_RGB_and_BGR_test():
    mt.get_function_name(ack=True, tabs=0)
    color_str = 'orange'
    rgb = pyplt.get_rgb_color(color_str)
    bgr = pyplt.get_bgr_color(color_str)

    rgba = pyplt.get_rgba_color(color_str)
    rgb2 = pyplt.rgba_to_rgb(rgba)
    bgr2 = pyplt.rgba_to_bgr(rgba)

    print('\tRGBA {} - {}'.format(rgba, color_str))
    print('\tRGB {}=={} ? {}'.format(rgb, rgb2, rgb == rgb2))
    print('\tBGR {}=={} ? {}'.format(bgr, bgr2, bgr == bgr2))
    return


def BGR_to_RGB_and_RGBA_test():
    mt.get_function_name(ack=True, tabs=0)
    color_str = 'orange'
    rgb = pyplt.get_rgb_color(color_str)
    rgba = pyplt.get_rgba_color(color_str)

    bgr = pyplt.get_bgr_color(color_str)
    rgb2 = pyplt.bgr_to_rgb(bgr)
    rgba2 = pyplt.bgr_to_rgba(bgr)
    print('\tBGR {} - {}'.format(bgr, color_str))
    print('\tRGB {}=={} ? {}'.format(rgb, rgb2, rgb == rgb2))
    print('\tRGBA {}=={} ? {}'.format(rgba, rgba2, rgba == rgba2))
    return


def RGB_to_RGBA_and_BGR_test():
    mt.get_function_name(ack=True, tabs=0)
    color_str = 'orange'
    bgr = pyplt.get_bgr_color(color_str)
    rgba = pyplt.get_rgba_color(color_str)

    rgb = pyplt.get_rgb_color(color_str)
    bgr2 = pyplt.rgb_to_bgr(rgb)
    rgba2 = pyplt.rgb_to_rgba(rgb)
    print('\tRGB {} - {}'.format(rgb, color_str))
    print('\tRGBA {}=={} ? {}'.format(rgba, rgba2, rgba == rgba2))
    print('\tBGR {}=={} ? {}'.format(bgr, bgr2, bgr == bgr2))
    return


def colors_plot_test():
    mt.get_function_name(ack=True, tabs=0)
    pyplt.colors_plot(ack=True, tabs=0, max_time=PLT_MAX_TIME)
    return


def get_ticks_list_test():
    mt.get_function_name(ack=True, tabs=0)
    ticks_list = pyplt.get_ticks_list(x_low=10, x_high=30, p=0.1)
    print(mt.to_str(ticks_list, '\tticks_list', chars=0))
    ticks_list = pyplt.get_ticks_list(x_low=0, x_high=500, p=0.2)
    print(mt.to_str(ticks_list, '\tticks_list', chars=0))
    return


def get_random_RGBA_color_map_test():
    mt.get_function_name(ack=True, tabs=0)
    random_RBGA_colors = pyplt.get_random_rgba_color_map(n=3)
    print(mt.to_str(random_RBGA_colors, '\trandom_color_map'))
    return


def screen_dims_test():
    mt.get_function_name(ack=True, tabs=0)
    sd = pyplt.screen_dims()
    print('\tscreen dims {}'.format(sd))
    return


def move_figure_x_y_test():
    mt.get_function_name(ack=True, tabs=0)
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    options = [(0, 0), (100, 0), (0, 100), (150, 150), (400, 400), (250, 350)]
    print('\tVisual test: move to all options {}'.format(options))
    print('\t\tClick Esc to close all')
    timers = []
    for x_y in options:
        title = 'move to {}'.format(x_y)
        fig, ax = plt.subplots()
        pyplt.set_window_title(fig, title='move_figure_x_y_test()')
        pyplt.set_figure_title(fig, title='move_figure_x_y_test()')
        pyplt.set_axes_title(ax, title)
        fig.canvas.mpl_connect('key_press_event', pyplt.on_click_close_all)
        ax.plot(t, s, c=pyplt.get_random_color('str'))
        pyplt.move_figure_x_y(fig, x_y=x_y)
        timer = pyplt.add_timer(fig, PLT_MAX_TIME)
        timers.append(timer)
    plt.show(block=True)
    for timer in timers:
        timer.stop()
    return


def move_figure_by_str_test():
    mt.get_function_name(ack=True, tabs=0)
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    options = pyplt.Location.get_location_list_by_rows()
    print('\tVisual test: move to all options {}'.format(options))
    print('\t\tClick Esc to close all')
    timers = []
    for where_to in options:
        title = 'move to {}'.format(where_to)
        resize = 0.7
        figsize = (6.4 * resize, 4.8 * resize)
        fig, ax = plt.subplots(figsize=figsize)
        pyplt.set_window_title(fig, title=title)
        pyplt.set_figure_title(fig, title=title)
        pyplt.set_axes_title(ax, title)
        fig.canvas.mpl_connect('key_press_event', pyplt.on_click_close_all)
        ax.plot(t, s, c=pyplt.get_random_color('str'))
        pyplt.move_figure_by_str(fig, where=where_to)
        timer = pyplt.add_timer(fig, PLT_MAX_TIME)
        timers.append(timer)
    plt.show(block=True)
    for timer in timers:
        timer.stop()
    return


def move_figure_by_short_str_test():
    mt.get_function_name(ack=True, tabs=0)
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    options = ['tl', 'cc', 'br']
    print('\tVisual test: move to all options {}'.format(options))
    print('\t\tClick Esc to close all')
    timers = []
    for where_to in options:
        title = 'move to {}'.format(where_to)
        resize = 0.7
        figsize = (6.4 * resize, 4.8 * resize)
        fig, ax = plt.subplots(figsize=figsize)
        pyplt.set_window_title(fig, title=title)
        pyplt.set_figure_title(fig, title=title)
        pyplt.set_axes_title(ax, title)
        fig.canvas.mpl_connect('key_press_event', pyplt.on_click_close_all)
        ax.plot(t, s, c=pyplt.get_random_color('str'))
        pyplt.move_figure_by_str(fig, where=where_to)
        timer = pyplt.add_timer(fig, PLT_MAX_TIME)
        timers.append(timer)
    plt.show(block=True)
    for timer in timers:
        timer.stop()
    return


def add_update_2d_scatter_test():
    mt.get_function_name(ack=True, tabs=0)
    fig, axes = plt.subplots()
    pyplt.set_window_title(fig, 'add_update_2d_scatter_test()')
    # create the fig world on 0 to 4 square
    data = [[0, 0], [4, 4]]  # nx2 base data. if you want a dashboard: data = [0, 0] - just set fig lims
    # c = 'g'  # equivalent: ['g'], pyplt.get_RGBA_color('g'), [pyplt.get_RGBA_color('g')]
    c = ['g', 'r']  # equivalent: [pyplt.get_RGBA_color('g'), 'r']
    sc = pyplt.add_2d_scatter(ax=axes, data=data, c=c, marker='o', marker_s=100, label='sc')
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    new_data = np.array([[2, 2], [3, 3]])  # notice data has to be in the fig world
    # colors = [pyplt.get_RGBA_color('y')]  # list of size 1 or n of RGBA colors
    # colors = pyplt.get_random_RBGA_color_map(n=2)
    colors = [pyplt.get_random_color('RGBA'), pyplt.get_rgba_color('b')]  # list of size 1 or n of RGBA colors
    pyplt.update_2d_scatter(sc=sc, sc_dict={'data': new_data, 'c': colors})
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    new_data = np.array([[1, 2.33], [3.4, 4.4]])  # notice data has to be in the fig world
    pyplt.update_2d_scatter(sc=sc, sc_dict={'data': new_data})  # no colors
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    pyplt.update_2d_scatter(sc=sc, sc_dict={'data': []})  # no colors

    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    return


def plot_2d_one_figure_iterative_test():
    mt.get_function_name(ack=True, tabs=0)
    iters = 2
    print('\tVisual test: {} iters of plot_2d_iterative_plot_test'.format(iters))

    data_plot_0 = {
        'title': 'main_plot',
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [0, 500, 0, 500],
        'xticks': pyplt.get_ticks_list(x_low=0, x_high=500, p=0.5),
        'datum': [
            {
                'data': mt.np_random_integers(low=0, high=480, size=(20, 2)),
                'c': 'r',
                'label': 'plot0_data0',
                'marker': 'x',
                'marker_s': 1,
            },
            {
                'data': mt.np_random_integers(low=0, high=480, size=(10, 2)),
                'c': pyplt.get_random_color('str'),
                'label': 'plot0_data1',
                'marker': 'o',
                'marker_s': 100,
            }
        ],
    }

    fig, axes, scatters_plot_i = pyplt.plot_2d_one_figure_iterative_init(
        fig_d={
            'title': 'plot_2d_one_figure_iterative_test()',
            'render': True,
            'block': False,
        },
        data_plot_i=data_plot_0,
        win_d={
            'title': 'plot_2d_one_figure_iterative_test()',
            'location': pyplt.Location.CENTER_CENTER.value,
            'resize': 1.2,
            'zoomed': False,
        },
    )

    for i in range(iters):
        mt.sleep(seconds=SLEEP)

        # emulate data of each scatter
        points_per_sc = 20

        new_data_plot_i = []
        for scatter in scatters_plot_i:  # for each scatter in sub plot
            # NOTICE we need same distribution data since the plots are not going to resize the axes
            # noinspection PyProtectedMember
            lows, highs = mt.get_uniform_dist_by_dim(scatter._offsets)
            new_data_i = {
                # 'data': None,  # no refresh
                # 'data': [],  # no refresh
                'data': mt.np_uniform(shape=(points_per_sc, 2), lows=lows, highs=highs),
                # 'c': None,  # color doesn't have to change
                # 'c': [pyplt.get_RGBA_color('g')]  # color could be a list of a color
                'c': pyplt.get_random_rgba_color_map(points_per_sc)  # color could be a list of n color
            }
            new_data_plot_i.append(new_data_i)

        pyplt.plot_2d_one_figure_iterative_update(
            fig_d={
                'fig': fig,
                'scatters': scatters_plot_i,
                'new_title': 'iter {}'.format(i),
                'render': True,
                'block': False,
            },
            new_data_plot_i=new_data_plot_i,
        )

        # second option if you have more stuff to render
        # pyplt.render_plot(fig, block=block)

    mt.create_dir(mtt.TEMP_FOLDER1)
    save_path = '{}/{}'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    pyplt.save_plot(path=save_path, ack=True, tabs=2)
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def plot_2d_many_figures_iterative_test():
    mt.get_function_name(ack=True, tabs=0)
    iters = 3
    print('\tVisual test: {} iters of plot_2d_iterative_plots'.format(iters))

    data_plot_0 = {
        'title': 'sub plot 0',
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [0, 500, 0, 500],
        'xticks': pyplt.get_ticks_list(x_low=0, x_high=500, p=0.5),
        'yticks': None,
        'datum': [
            {
                'data': mt.np_random_integers(low=0, high=480, size=(20, 2)),
                'c': 'r',
                'label': 'plot0_data0',
                'marker': 'x',
                'marker_s': 100,
            },
            {
                'data': mt.np_random_integers(low=0, high=480, size=(10, 2)),
                'c': pyplt.get_random_color('str'),
                'label': 'plot0_data1',
                'marker': pyplt.get_random_marker(),
                'marker_s': 100,
            }
        ],
    }
    data_plot_1 = {
        'title': 'sub plot 1',
        'legend': {
            'loc': 'lower center',
            'ncol': 1,
            'fancybox': True,
            'framealpha': 0.5,
            'edgecolor': 'black'
        },
        'fig_lims': None,
        'xticks': None,
        'yticks': None,
        'datum': [
            {
                'data': mt.np_uniform(lows=0, highs=0.03, shape=(20, 2)),
                'c': 'magenta',
                'label': 'plot1_data0',
                'marker': '.',
            },
        ],
    }
    data_plot_2 = {  # minimum fields
        'datum': [
            {
                'data': mt.np_normal(mius=0, stds=0.5, shape=(20, 2)),
            },
        ],
    }
    data_plot_3 = {
        'title': 'sub plot 3',
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': None,
        'xticks': None,
        'yticks': None,
        'datum': [
            {
                'data': mt.np_uniform(lows=0, highs=0.03, shape=(20, 2)),
                'c': None,
                'label': 'plot3_data0',
                'marker': None,
            },
        ],
    }

    fig, axes_list, scatters = pyplt.plot_2d_many_figures_iterative_init(
        fig_d={
            'title': 'fig_title',
            'grid_xy': (2, 2),
            'render': True,
            'block': False,
        },
        data_all_plots=[data_plot_0, data_plot_1, data_plot_2, data_plot_3],
        win_d={
            'title': 'plot_2d_many_figures_iterative_test()',
            'location': pyplt.Location.TOP_CENTER.value,
            'resize': 1.5,
            'zoomed': False,
        },
    )

    for i in range(iters):
        mt.sleep(seconds=SLEEP)

        # emulate data of each scatter
        points_per_sc = 20
        new_data_all_plots = []
        for scatters_plot_i in scatters:  # for each sub plot
            new_data_plot_i = []
            for scatter in scatters_plot_i:  # for each scatter in sub plot
                # NOTICE we need same distribution data since the plots are not going to resize the axes
                # noinspection PyProtectedMember
                lows, highs = mt.get_uniform_dist_by_dim(scatter._offsets)
                new_data_i = {
                    # 'data': None,  # no refresh
                    # 'data': [],  # no refresh
                    'data': mt.np_uniform(shape=(points_per_sc, 2), lows=lows, highs=highs),
                    # 'c': None,  # color doesn't have to change
                    # 'c': [pyplt.get_RGBA_color('g')]  # color could be a list of a color
                    'c': pyplt.get_random_rgba_color_map(points_per_sc)  # color could be a list of n color
                }
                new_data_plot_i.append(new_data_i)
            new_data_all_plots.append(new_data_plot_i)

        pyplt.plot_2d_many_figures_iterative_update(
            fig_d={
                'fig': fig,
                'scatters': scatters,
                'new_title': 'iter {}'.format(i),
                'render': True,
                'block': False,
            },
            new_data_all_plots=new_data_all_plots,
        )

        # second option if you have more stuff to render
        # pyplt.render_plot(fig, block=block)
    mt.create_dir(mtt.TEMP_FOLDER1)
    save_path = '{}/{}'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    pyplt.save_plot(path=save_path, ack=True, tabs=2)
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def plot_2d_one_figure_test():
    mt.get_function_name(ack=True, tabs=0)
    datum = [
        {
            'data': mt.np_random_integers(low=0, high=480, size=(20, 2)),
            'c': pyplt.get_random_color('str'),
            'label': 'plot0_data0',
            'marker': 'x',
        },
        {
            'data': mt.np_random_integers(low=0, high=480, size=(10, 2)),
            'c': 'g',
            'label': 'plot0_data1',
            'marker': 'o',
        }
    ]

    print('\tVisual test: plot_2d_scatter_test')
    pyplt.plot_2d_one_figure(
        datum=datum,
        fig_title='plot_2d_one_figure_test()',
        win_d=pyplt.WINDOW_DEFAULT,
        max_time=PLT_MAX_TIME
    )
    return


def plot_2d_many_figures_test():
    mt.get_function_name(ack=True, tabs=0)
    datum_i = [
        {
            'data': mt.np_random_integers(low=0, high=480, size=(20, 2)),
            'c': 'r',
            'label': 'plot0_data0',
            'marker': 'x',
        },
        {
            'data': mt.np_random_integers(low=0, high=480, size=(10, 2)),
            'c': 'g',
            'label': 'plot0_data1',
            'marker': 'o',
        }
    ]

    datum_list = [datum_i, datum_i]

    print('\tVisual test: plot_2d_scatters_test')
    pyplt.plot_2d_many_figures(
        grid=(1, 2),
        datum_list=datum_list,
        sub_titles=['t1', 't2'],
        fig_title='plot_2d_many_figures_test()',
        win_d=pyplt.WINDOW_DEFAULT,
        max_time=PLT_MAX_TIME
    )
    return


def plot_2d_iterative_dashboard_test():
    mt.get_function_name(ack=True, tabs=0)
    iters = 2
    print('\tVisual test: {} iters of plot_2d_iterative_dashboard_test'.format(iters))

    data_plot_0 = {
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [0, 500, 0, 500],
        'xticks': None,
        'yticks': None,
        'datum': [
            {
                'c': pyplt.get_rgba_color('r'),
                'label': 'plot0_data0',
                'marker': pyplt.get_random_marker(),
                'marker_s': 125,
            },
            {
                'c': 'g',
                'label': 'plot0_data1',
                'marker': pyplt.get_random_marker(),
            }
        ],
    }

    fig, axes, scatters_plot_i = pyplt.plot_2d_iterative_dashboard_init(
        fig_d={
            'title': 'fig_title',
            'render': True,
            'block': False,
        },
        data_plot_i=data_plot_0,
        win_d={
            'title': 'plot_2d_iterative_dashboard_test()',
            'location': pyplt.Location.BOTTOM_CENTER.value,
            'resize': 1.2,
            'zoomed': False,
        },
        center_d=pyplt.CENTER_DEFAULT,
    )

    for i in range(iters):
        mt.sleep(seconds=SLEEP)

        # emulate data of each scatter
        points_per_sc = 20

        new_data_plot_i = []
        for _ in scatters_plot_i:  # for each scatter in sub plot
            new_data_i = {
                # 'data': None,  # no refresh
                # 'data': [],  # no refresh
                'data': mt.np_random_integers(low=0, high=480, size=(points_per_sc, 2)),
                # 'c': None,  # color doesn't have to change
                # 'c': [pyplt.get_RGBA_color('g')]  # color could be a list of a RGBA color
                # 'c': pyplt.get_random_RBGA_color_map(points_per_sc),  # color could be a list of n RGBA color
            }
            new_data_plot_i.append(new_data_i)

        pyplt.plot_2d_one_figure_iterative_update(
            fig_d={
                'fig': fig,
                'scatters': scatters_plot_i,
                'new_title': 'iter {}'.format(i),
                'render': True,
                'block': False,
            },
            new_data_plot_i=new_data_plot_i,
        )

        # second option if you have more stuff to render
        # pyplt.render_plot(fig, block=block)
    mt.create_dir(mtt.TEMP_FOLDER1)
    save_path = '{}/{}'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    pyplt.save_plot(path=save_path, ack=True, tabs=2)
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def plot_2d_iterative_dashboards_test():
    mt.get_function_name(ack=True, tabs=0)
    iters = 3
    print('\tVisual test: {} iters of plot_2d_iterative_plots'.format(iters))

    data_plot_0 = {
        'title': 'sub plot 0',
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [0, 500, 0, 500],
        'xticks': pyplt.get_ticks_list(x_low=0, x_high=500, p=0.5),
        'datum': [
            {
                'label': 'plot0_data0',
                'marker_s': 120,
            },
            {
                'c': 'g',
                'label': 'plot0_data1',
                'marker': pyplt.get_random_marker(),
            }
        ],
    }
    data_plot_1 = {
        'title': 'sub plot 1',
        'fig_lims': [0, 500, 0, 500],
        'datum': [
            {
                'c': 'magenta',
                'label': 'plot1_data0',
                'marker': pyplt.get_random_marker(),
            },
        ],
    }

    fig, axes_list, scatters = pyplt.plot_2d_iterative_dashboards_init(
        fig_d={
            'title': 'fig_title',
            'grid_xy': (1, 2),
            'render': True,
            'block': False,
        },
        data_all_plots=[data_plot_0, data_plot_1],
        win_d={
            'title': 'plot_2d_iterative_dashboards_test()',
            'location': pyplt.Location.TOP_CENTER.value,
            'resize': 1.5,
            'zoomed': False,
        },
        center_d=pyplt.CENTER_DEFAULT,
    )

    for i in range(iters):
        mt.sleep(seconds=SLEEP)

        # emulate data of each scatter
        points_per_sc = 20
        new_data_all_plots = []
        for scatters_plot_i in scatters:  # for each sub plot
            new_data_plot_i = []
            for _ in scatters_plot_i:  # for each scatter in sub plot
                new_data_i = {
                    # 'data': None,  # no refresh
                    # 'data': [],  # no refresh
                    'data': mt.np_random_integers(low=0, high=480, size=(20, 2)),
                    # 'c': None,  # color doesn't have to change
                    # 'c': [pyplt.get_RGBA_color('g')]  # color could be a list of a color
                    'c': pyplt.get_random_rgba_color_map(points_per_sc)  # color could be a list of n color
                }
                new_data_plot_i.append(new_data_i)
            new_data_all_plots.append(new_data_plot_i)

        pyplt.plot_2d_many_figures_iterative_update(
            fig_d={
                'fig': fig,
                'scatters': scatters,
                'new_title': 'iter {}'.format(i),
                'render': True,
                'block': False,
            },
            new_data_all_plots=new_data_all_plots,
        )

        # second option if you have more stuff to render
        # pyplt.render_plot(fig, block=block)
    mt.create_dir(mtt.TEMP_FOLDER1)
    save_path = '{}/{}'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    pyplt.save_plot(path=save_path, ack=True, tabs=2)
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def plot_x_y_std_test():
    mt.get_function_name(ack=True, tabs=0)
    data_x = [10, 20, 30]

    C_errors = [5, 7, 1]
    C_errors_stds = [2, 1, 0.5]
    group_c = (C_errors, C_errors_stds, 'g', 'C')

    U_errors = [10, 8, 3]
    U_errors_vars = [4, 3, 1.5]
    group_u = (U_errors, U_errors_vars, pyplt.get_random_color(), 'U')

    print('\tVisual test: errors and stds')
    pyplt.plot_x_y_std(
        data_x,
        groups=[group_c, group_u],
        title='plot_x_y_std_test()',
        legend=pyplt.LEGEND_DEFAULT,
        x_label='X',
        y_label='Y',
        save_path=None,
        show_plot=True,
        with_shift=True,
        max_time=PLT_MAX_TIME
    )
    return


def histogram_test():
    mt.get_function_name(ack=True, tabs=0)
    data = mt.np_uniform(shape=(1000,), lows=0, highs=10000)
    print('\tVisual test: histogram')
    pyplt.histogram(
        values=data,
        title='histogram: 10 bins of 1000 numbers from 0 to 10000',
        save_path=None,
        bins_n=10,
        max_time=PLT_MAX_TIME
    )
    return


def compare_images_sets_test():
    mt.get_function_name(ack=True, tabs=0)
    try:
        from wizzi_utils.open_cv import open_cv_tools as cvt
        from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
        print('\tVisual test: 2 compare_images_sets')
        title = 'compare_images_sets_test()'
        imgBGR = cvtt.load_img_from_web(mtt.SO_LOGO)
        imgRGB = cvt.bgr_to_rgb(imgBGR)
        img_gray = cvt.bgr_to_gray(imgBGR)
        img_gray = cvt.gray_to_bgr(img_gray)
        # print(mt.to_str(imgBGR))
        # print(mt.to_str(imgRGB))
        # print(mt.to_str(img_gray))
        img_set_a = np.array([imgRGB, imgRGB, imgRGB], dtype=np.uint8)
        img_set_b = np.array([imgRGB, imgBGR, img_gray], dtype=np.uint8)
        # print(mt.to_str(img_set_a))
        # print(mt.to_str(img_set_b))
        pyplt.compare_images_sets(img_set_a, img_set_b, title=title, max_time=PLT_MAX_TIME)
    except ModuleNotFoundError as e:
        mt.exception_error(e, real_exception=True)
    return


def compare_images_multi_sets_squeezed_test():
    mt.get_function_name(ack=True, tabs=0)
    try:
        from wizzi_utils.open_cv import open_cv_tools as cvt
        from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
        # # OLD CODE
        # import torch
        # from torchvision import datasets
        # import torchvision.transforms as transforms
        # transform = transforms.Compose([transforms.ToTensor(), ])
        # # choose data set - both work
        # # data_root = path to the data else download
        # print('\tVisual test: 2 compare_images_multi_sets_squeezed')
        # data_root = './Datasets/'
        # mt.create_dir(data_root, ack=True)
        # # dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
        # dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
        # # noinspection PyUnresolvedReferences
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        # # noinspection PyUnresolvedReferences
        # images32, labels = iter(data_loader).next()
        #
        # images = images32[:16]  # imagine the first 16 are base images and predicted_images are the model output
        # predicted_images = images32[16:32]
        # print(images.shape)
        # print(predicted_images.shape)
        # d = {'original_data': images, 'predicted_data': predicted_images}
        # pyplt.compare_images_multi_sets_squeezed(
        #     sets_dict=d,
        #     title='comp',
        #     desc=True,
        #     tabs=1
        # )

        print('\tVisual test: 2 compare_images_multi_sets_squeezed')
        title = 'compare_images_multi_sets_squeezed_test()'
        imgBGR = cvtt.load_img_from_web(mtt.SO_LOGO)
        imgRGB = cvt.bgr_to_rgb(imgBGR)
        img_gray = cvt.bgr_to_gray(imgBGR)
        img_gray = cvt.gray_to_bgr(img_gray)
        # print(mt.to_str(imgBGR))
        # print(mt.to_str(imgRGB))
        # print(mt.to_str(img_gray))

        # from (height,width,3) -> (3,height,width)
        imgBGR = np.swapaxes(imgBGR, 1, 2)
        imgBGR = np.swapaxes(imgBGR, 0, 1)
        imgRGB = np.swapaxes(imgRGB, 1, 2)
        imgRGB = np.swapaxes(imgRGB, 0, 1)
        img_gray = np.swapaxes(img_gray, 1, 2)
        img_gray = np.swapaxes(img_gray, 0, 1)
        # print(mt.to_str(imgBGR))
        # print(mt.to_str(imgRGB))
        # print(mt.to_str(img_gray))

        # need to normalize by div max color
        img_set_a = np.array([imgRGB, imgRGB, imgRGB], dtype=np.float) / 255
        img_set_b = np.array([imgRGB, imgBGR, img_gray], dtype=np.float) / 255
        # print(mt.to_str(img_set_a))
        # print(mt.to_str(img_set_b))

        d = {
            'original_data': img_set_a,
            'predicted_data': img_set_b,
            'for_fun': img_set_b,
            'for_fun2': img_set_b
        }
        pyplt.compare_images_multi_sets_squeezed(
            sets_dict=d,
            title=title,
            ack=True,
            tabs=1,
            max_time=PLT_MAX_TIME
        )

    except ModuleNotFoundError as e:
        mt.exception_error(e, real_exception=True)
    return


def add_update_3d_scatter_test():
    mt.get_function_name(ack=True, tabs=0)
    fig = plt.figure()
    axes = Axes3D(fig)
    pyplt.set_window_title(fig, 'add_update_2d_scatter_test()')
    # create the fig world on 0 to 5 cube
    # nx3 base data. if you want a dashboard: data = [0, 0,0] - just set fig lims
    data = [[0, 0, 0], [4, 4, 4], [5, 5, 5]]
    # c = 'g'  # equivalent: ['g'], pyplt.get_RGBA_color('g'), [pyplt.get_RGBA_color('g')]
    c = ['g', 'r', 'b']  # equivalent: [pyplt.get_RGBA_color('g'), 'r', 'b]
    sc = pyplt.add_3d_scatter(ax=axes, data=data, c=c, marker='o', marker_s=100, label='sc')
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    new_data = np.array([[2, 2, 2], [3, 3, 3]])  # notice data has to be in the fig world
    # colors = 'y' # equivalent pyplt.get_RGBA_color('y'), [pyplt.get_RGBA_color('y')]
    colors = ['y', pyplt.get_rgba_color('b')]  # list of size n of RGBA or str colors
    pyplt.update_3d_scatter(sc=sc, sc_dict={'data': new_data, 'c': colors})
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    new_data = np.array([[1, 2.33, 2.1], [3.4, 4.4, 3.1]])  # notice data has to be in the fig world
    pyplt.update_3d_scatter(sc=sc, sc_dict={'data': new_data})  # no colors
    pyplt.render_plot(fig=fig, block=False)

    mt.sleep(seconds=SLEEP)
    pyplt.update_3d_scatter(sc=sc, sc_dict={'data': []})  # no colors

    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    return


def plot_3d_iterative_dashboard_test():
    mt.get_function_name(ack=True, tabs=0)
    iters = 3
    print('\tVisual test: {} iters of 3d iterative plot'.format(iters))

    data_plot_0 = {
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [-20, 20, -20, 20, -20, 20],
        'color_axes': 'b',
        'view': {'azim': -60, 'elev': 30},
        'face_color': 'white',
        'background_color': 'white',
        'xticks': None,
        'yticks': None,
        'zticks': pyplt.get_ticks_list(-20, 20, 0.5),
        'datum': [
            {
                'c': pyplt.get_rgba_color('r'),
                'label': 'plot0_data0',
                'marker': 'o',
                'marker_s': 150,
            },
            {
                'c': 'g',
                'label': 'plot0_data1',
                'marker': pyplt.get_random_marker(),
                'marker_s': 20,
            }
        ]
    }

    fig, ax, iterative_scatters = pyplt.plot_3d_iterative_dashboard_init(
        fig_d={
            'title': 'iter -1',
            'render': False,
            'block': False,

        },
        data_plot_i=data_plot_0,
        win_d={
            'title': 'plot_3d_iterative_dashboard_test()',
            'location': pyplt.Location.CENTER_CENTER.value,
            'resize': 1.5,
            'zoomed': False,
        },
        center_d=pyplt.CENTER_DEFAULT,
    )

    # # CUSTOM ADD ON 1 - update each round
    center_mass_x_y = {"x1": 0.05, "y1": 0.95, }
    center_mass_label_base = "data0:(xyz)={}"
    center_mass_label = ax.text2D(
        x=center_mass_x_y['x1'],
        y=center_mass_x_y['y1'],
        s=center_mass_label_base.format(np.zeros(3)),
        transform=ax.transAxes,
        color='green'
    )

    # # CUSTOM ADD ON 2 - done once
    pyplt.add_cube3d_around_origin(ax, edge_len=4, add_labels=False)

    # should call legend if you want custom add ons in it
    legend = pyplt.LEGEND_DEFAULT
    plt.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
               framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])

    pyplt.render_plot(fig=fig, block=False)  # not necessary - only if you want to see the plot before first data comes
    points_per_sc = 20

    for i in range(iters):
        mt.sleep(seconds=SLEEP)

        # emulate data of each scatter
        new_data_plot_i = []
        for _ in iterative_scatters:  # for each scatter in sub plot
            new_data_i = {
                # 'data': None,  # no refresh
                # 'data': [],  # no refresh
                'data': mt.np_random_integers(low=-15, high=15, size=(points_per_sc, 3)),
                # 'c': None,  # color doesn't have to change
                # 'c': pyplt.get_RGBA_color('r')  # color could be a RGBA color
                # 'c': [pyplt.get_RGBA_color('g')]  # color could be a list of a RGBA color
                'c': pyplt.get_random_rgba_color_map(points_per_sc),  # color could be a list of n RGBA color
                # 'c': 'r'  # color could be a single str color
                # 'c': 'red'  #  color could be a single str color
                # 'c': ['red'] * points_per_sc  # color could be a list of n string colors
            }
            new_data_plot_i.append(new_data_i)

        pyplt.plot_3d_one_figure_iterative_update(
            fig_d={
                'fig': fig,
                'scatters': iterative_scatters,
                'new_title': 'iter {}'.format(i),
                'render': False,
                'block': False,
            },
            new_data_plot_i=new_data_plot_i,
        )

        # # CUSTOM ADD ON 1 - update each round
        center_mass_label.set_text(
            center_mass_label_base.format(np.round(np.mean(new_data_plot_i[0]['data'], axis=0), 2)))
        pyplt.render_plot(fig, block=False)  # must render here due to custom add on
    mt.create_dir(mtt.TEMP_FOLDER1)
    save_path = '{}/{}'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    pyplt.save_plot(path=save_path, ack=True, tabs=2)
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def plot_3d_one_dashboard():
    mt.get_function_name(ack=True, tabs=0)
    points_per_sc = 10
    data_plot_0 = {
        'legend': pyplt.LEGEND_DEFAULT,
        'fig_lims': [-20, 20, -20, 20, -20, 20],
        'color_axes': 'b',
        'view': {'azim': -60, 'elev': 30},
        'face_color': 'white',
        'background_color': 'white',
        'xticks': None,
        'yticks': None,
        'zticks': pyplt.get_ticks_list(-20, 20, 0.5),
        'datum': [
            {
                'data': mt.np_random_integers(low=-15, high=15, size=(points_per_sc, 3)),
                'c': 'r',
                'label': 'plot0_data0',
                'marker': 'o',
                'marker_s': 150,
            },
            {
                'data': mt.np_random_integers(low=-15, high=15, size=(points_per_sc, 3)),
                'c': 'g',
                'label': 'plot0_data1',
                'marker': pyplt.get_random_marker(),
                'marker_s': 20,
            }
        ]
    }

    print('\tVisual test: scatter 3d plot')

    pyplt.plot_3d_one_dashboard(
        fig_title='plot_3d_one_dashboard()',
        data_plot_i=data_plot_0,
        win_d={
            'title': 'plot_3d_one_dashboard()',
            'location': pyplt.Location.CENTER_CENTER.value,
            'resize': 1.5,
            'zoomed': False,
        },
        center_d=pyplt.CENTER_DEFAULT,
        max_time=PLT_MAX_TIME
    )

    return


def plot_3d_cube_test():
    mt.get_function_name(ack=True, tabs=0)
    fig = plt.figure()
    pyplt.set_figure_title(fig, title='plot_3d_cube_test()')
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # base on 0,0,0 and get points from left right and top by edge
    edge = 10.0
    point_base = np.array([0., 0., 0.])
    point_edge_left = np.array([0., edge, 0.])
    point_edge_right = np.array([edge, 0., 0.])
    point_edge_top = np.array([0., 0., edge])
    cube_def = [point_base, point_edge_left, point_edge_right, point_edge_top]

    pyplt.plot_3d_cube(
        ax,
        cube_definition=cube_def,
        label='my cube',
        face_color='blue',
        face_opacity=0.5,
        edge_color='black',
        edge_width=3.0,
        add_corners_coordinates=True,
    )
    legend = pyplt.LEGEND_DEFAULT
    plt.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
               framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])

    print('\tVisual test: plot_3d_cube_test')
    pyplt.move_figure_by_str(plt.gcf())
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    return


def add_cube3d_around_origin_test():
    mt.get_function_name(ack=True, tabs=0)
    fig = plt.figure()
    pyplt.set_figure_title(fig, title='add_cube3d_around_origin_test()')
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pyplt.add_cube3d_around_origin(
        ax,
        edge_len=4,
        color='green',
        add_labels=False
    )

    legend = pyplt.LEGEND_DEFAULT
    plt.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
               framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])

    print('\tVisual test: add_cube3d_around_origin')
    pyplt.finalize_plot(fig, PLT_MAX_TIME)
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    get_colors_formats_test()
    RGBA_to_RGB_and_BGR_test()
    BGR_to_RGB_and_RGBA_test()
    RGB_to_RGBA_and_BGR_test()
    get_ticks_list_test()
    colors_plot_test()
    get_random_RGBA_color_map_test()
    screen_dims_test()
    move_figure_x_y_test()
    move_figure_by_str_test()
    move_figure_by_short_str_test()

    # iterative plots - 2d and 3d
    add_update_2d_scatter_test()
    plot_2d_many_figures_iterative_test()
    plot_2d_one_figure_iterative_test()
    plot_2d_iterative_dashboards_test()
    plot_2d_iterative_dashboard_test()
    add_update_3d_scatter_test()
    plot_3d_iterative_dashboard_test()

    # 2d plots
    plot_2d_one_figure_test()
    plot_2d_many_figures_test()
    plot_x_y_std_test()
    histogram_test()
    # 2d comparision
    compare_images_sets_test()
    compare_images_multi_sets_squeezed_test()

    # 3d plots
    plot_3d_one_dashboard()
    plot_3d_cube_test()
    add_cube3d_around_origin_test()

    print('{}'.format('-' * 20))
    return
