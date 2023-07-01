import numpy as np
import sys
import math
from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.backend_bases import CloseEvent
from matplotlib.collections import PathCollection
from wizzi_utils.misc import misc_tools as mt


class Location(Enum):
    TOP_LEFT = 'top_left'
    TOP_CENTER = 'top_center'
    TOP_RIGHT = 'top_right'
    CENTER_LEFT = 'center_left'
    CENTER_CENTER = 'center_center'
    CENTER_RIGHT = 'center_right'
    BOTTOM_LEFT = 'bottom_left'
    BOTTOM_CENTER = 'bottom_center'
    BOTTOM_RIGHT = 'bottom_right'

    @staticmethod
    def where_to(w_short: str) -> str:
        """
        instead of writing:
        wu.pyplt.Location.TOP_LEFT.value
        write:
        wu.pyplt.Location.where('tl')
        :param w_short:
        :return:
        """
        w_long = None
        if w_short == 'tl':
            w_long = Location.TOP_LEFT.value
        elif w_short == 'tc':
            w_long = Location.TOP_CENTER.value
        elif w_short == 'tr':
            w_long = Location.TOP_RIGHT.value
        elif w_short == 'cl':
            w_long = Location.CENTER_LEFT.value
        elif w_short == 'cc':
            w_long = Location.CENTER_CENTER.value
        elif w_short == 'cr':
            w_long = Location.CENTER_RIGHT.value
        elif w_short == 'bl':
            w_long = Location.BOTTOM_LEFT.value
        elif w_short == 'bc':
            w_long = Location.BOTTOM_CENTER.value
        elif w_short == 'br':
            w_long = Location.BOTTOM_RIGHT.value
        return w_long

    @staticmethod
    def get_location_list_by_rows():
        """
        options go by rows
            top left, top center, top right,
            center left ...
        """
        return [loc.value for loc in Location]

    @staticmethod
    def get_location_list_by_cols():
        """
        options go by cols
            top left, center left, bottom left,
            top center ...
        """
        col1 = [loc.value for loc in Location if loc.value.endswith('left')]
        col2 = [loc.value for loc in Location if loc.value.endswith('center')]
        col3 = [loc.value for loc in Location if loc.value.endswith('right')]
        li = col1 + col2 + col3
        return li


COLORS_STRINGS = [  # sorted by color map - see colors_plot()
    'black',  # 0
    'dimgray',  # 1
    'dimgrey',  # 2
    'gray',  # 3
    'grey',  # 4
    'darkgray',  # 5
    'darkgrey',  # 6
    'silver',  # 7
    'lightgray',  # 8
    'lightgrey',  # 9
    'gainsboro',  # 10
    'whitesmoke',  # 11
    'white',  # 12
    'snow',  # 13
    'rosybrown',  # 14
    'lightcoral',  # 15
    'indianred',  # 16
    'brown',  # 17
    'firebrick',  # 18
    'maroon',  # 19
    'darkred',  # 20
    'red',  # 21
    'mistyrose',  # 22
    'salmon',  # 23
    'tomato',  # 24
    'darksalmon',  # 25
    'coral',  # 26
    'orangered',  # 27
    'lightsalmon',  # 28
    'sienna',  # 29
    'seashell',  # 30
    'chocolate',  # 31
    'saddlebrown',  # 32
    'sandybrown',  # 33
    'peachpuff',  # 34
    'peru',  # 35
    'linen',  # 36
    'bisque',  # 37
    'darkorange',  # 38
    'burlywood',  # 39
    'antiquewhite',  # 40
    'tan',  # 41
    'navajowhite',  # 42
    'blanchedalmond',  # 43
    'papayawhip',  # 44
    'moccasin',  # 45
    'orange',  # 46
    'wheat',  # 47
    'oldlace',  # 48
    'floralwhite',  # 49
    'darkgoldenrod',  # 50
    'goldenrod',  # 51
    'cornsilk',  # 52
    'gold',  # 53
    'lemonchiffon',  # 54
    'khaki',  # 55
    'palegoldenrod',  # 56
    'darkkhaki',  # 57
    'ivory',  # 58
    'beige',  # 59
    'lightyellow',  # 60
    'lightgoldenrodyellow',  # 61
    'olive',  # 62
    'yellow',  # 63
    'olivedrab',  # 64
    'yellowgreen',  # 65
    'darkolivegreen',  # 66
    'greenyellow',  # 67
    'chartreuse',  # 68
    'lawngreen',  # 69
    'honeydew',  # 70
    'darkseagreen',  # 71
    'palegreen',  # 72
    'lightgreen',  # 73
    'forestgreen',  # 74
    'limegreen',  # 75
    'darkgreen',  # 76
    'green',  # 77
    'lime',  # 78
    'seagreen',  # 79
    'mediumseagreen',  # 80
    'springgreen',  # 81
    'mintcream',  # 82
    'mediumspringgreen',  # 83
    'mediumaquamarine',  # 84
    'aquamarine',  # 85
    'turquoise',  # 86
    'lightseagreen',  # 87
    'mediumturquoise',  # 88
    'azure',  # 89
    'lightcyan',  # 90
    'paleturquoise',  # 91
    'darkslategray',  # 92
    'darkslategrey',  # 93
    'teal',  # 94
    'darkcyan',  # 95
    'aqua',  # 96
    'cyan',  # 97
    'darkturquoise',  # 98
    'cadetblue',  # 99
    'powderblue',  # 100
    'lightblue',  # 101
    'deepskyblue',  # 102
    'skyblue',  # 103
    'lightskyblue',  # 104
    'steelblue',  # 105
    'aliceblue',  # 106
    'dodgerblue',  # 107
    'lightslategray',  # 108
    'lightslategrey',  # 109
    'slategray',  # 110
    'slategrey',  # 111
    'lightsteelblue',  # 112
    'cornflowerblue',  # 113
    'royalblue',  # 114
    'ghostwhite',  # 115
    'lavender',  # 116
    'midnightblue',  # 117
    'navy',  # 118
    'darkblue',  # 119
    'mediumblue',  # 120
    'blue',  # 121
    'slateblue',  # 122
    'darkslateblue',  # 123
    'mediumslateblue',  # 124
    'mediumpurple',  # 125
    'rebeccapurple',  # 126
    'blueviolet',  # 127
    'indigo',  # 128
    'darkorchid',  # 129
    'darkviolet',  # 130
    'mediumorchid',  # 131
    'thistle',  # 132
    'plum',  # 133
    'violet',  # 134
    'purple',  # 135
    'darkmagenta',  # 136
    'fuchsia',  # 137
    'magenta',  # 138
    'orchid',  # 139
    'mediumvioletred',  # 140
    'deeppink',  # 141
    'hotpink',  # 142
    'lavenderblush',  # 143
    'palevioletred',  # 144
    'crimson',  # 145
    'pink',  # 146
    'lightpink',  # 147
]

WINDOW_DEFAULT = {
    'title': None,
    'location': (0, 0),
    'resize': None,
    'zoomed': False
}
LEGEND_DEFAULT = {
    'loc': 'upper right',
    'ncol': 1,
    'fancybox': True,
    'framealpha': 0.5,
    'edgecolor': 'black'
}
CENTER_DEFAULT = {
    'c': 'orange',
    'marker': 'x',
    'marker_size': 150,
    'label': 'C'
}

MARKERS = [
    ".",  # m00	point
    ",",  # m01	pixel
    "o",  # m02	circle
    "v",  # m03	triangle_down
    "^",  # m04	triangle_up
    "<",  # m05	triangle_left
    ">",  # m06	triangle_right
    "1",  # m07	tri_down
    "2",  # m08	tri_up
    "3",  # m09	tri_left
    "4",  # m10	tri_right
    "8",  # m11	octagon
    "s",  # m12	square
    "p",  # m13	pentagon
    "P",  # m23	plus (filled)
    "*",  # m14	star
    "h",  # m15	hexagon1
    "H",  # m16	hexagon2
    "+",  # m17	plus
    "x",  # m18	x
    "X",  # m24	x (filled)
    "D",  # m19	diamond
    "d",  # m20	thin_diamond
    "|",  # m21	v line
    "_",  # m22	h line
    0,  # (TICK LEFT)	m25	tick left
    1,  # (TICK RIGHT)	m26	tick right
    2,  # (TICK UP)	m27	tick up
    3,  # (TICK DOWN)	m28	tick down
    4,  # (CARET LEFT)	m29	caret left
    5,  # (CARET RIGHT)	m30	caret right
    6,  # (CARET UP)	m31	care tup
    7,  # (CARET DOWN)	m32	caret down
    8,  # (CARET LEFT BASE)	m33	caret left (centered at base)
    9,  # (CARET RIGHT BASE)	m34	caret right (centered at base)
    10,  # (CARET UP BASE)	m35	caret up (centered at base)
    11,  # (CARET DOWN BASE)	m36	caretdown (centered at base)
]


def get_rgb_color(color_str: str) -> tuple:
    """
    :param color_str: e.g. "red", "blue" ...
        see COLORS_STRINGS for options
    :return: RGB color
    see get_colors_formats_test()
    """
    # noinspection PyUnresolvedReferences
    rgb_normed = matplotlib.colors.to_rgb(color_str)
    rgb = tuple([int(round(255 * x)) for x in rgb_normed])
    return rgb


def get_rgba_color(color_str: str, opacity: float = 1.0, fp: int = 3) -> tuple:
    """
    :param color_str: e.g. "red", "blue" ...
        see COLORS_STRINGS for options
    :param opacity: value from 0 to 1
    :param fp: float_pre>=0: round to x digits - None no rounding
    :return: rgba color
    see get_colors_formats_test()
    """
    # noinspection PyUnresolvedReferences
    rgba = matplotlib.colors.to_rgba(color_str, alpha=opacity)
    if fp is not None and fp >= 0:
        rgba = mt.round_tuple(rgba, fp)
    return rgba


def get_bgr_color(color_str: str) -> tuple:
    """
    :param color_str: e.g. "red", "blue" ...
        see COLORS_STRINGS for options
    :return: bgr color
    see get_colors_formats_test()
    """
    rgb = get_rgb_color(color_str)
    bgr = rgb_to_bgr(rgb)
    return bgr


def rgba_to_rgb(rgba: tuple) -> tuple:
    """
    :param rgba: RGBA format - tuple 1,4
    :return: rgb color format
    see RGBA_to_RGB_and_BGR_test()
    """
    # noinspection PyUnresolvedReferences
    rgb_normed = matplotlib.colors.to_rgb(rgba)
    rgb = tuple([int(round(255 * x)) for x in rgb_normed])
    return rgb


def rgba_to_bgr(rgba: tuple) -> tuple:
    """
    :param rgba: RGBA format - tuple 1,4
    :return: bgr color format
    see RGBA_to_RGB_and_BGR_test()
    """
    rgb = rgba_to_rgb(rgba)
    bgr = rgb_to_bgr(rgb)
    return bgr


def bgr_to_rgb(bgr: tuple) -> tuple:
    """
    :param bgr:
    :return: rgb
    see BGR_to_RGB_and_RGBA_test()
    """
    rgb = mt.reverse_tuple_or_list(bgr)  # reverse the tuple order
    return rgb


def bgr_to_rgba(bgr: tuple, opacity: float = 1.0) -> tuple:
    """
    :param bgr:
    :param opacity:
    :return: rgba
    see BGR_to_RGB_and_RGBA_test()
    """
    rgb = bgr_to_rgb(bgr)
    rgba = rgb_to_rgba(rgb, opacity)
    return rgba


def rgb_to_bgr(rgb: tuple) -> tuple:
    """
    :param rgb:
    :return: bgr
    see RGB_to_RGBA_and_BGR_test()
    """
    bgr = mt.reverse_tuple_or_list(rgb)  # reverse the tuple order
    return bgr


def rgb_to_rgba(rgb: tuple, opacity: float = 1.0, fp: int = 3) -> tuple:
    """
    :param rgb:
    :param opacity:
    :param fp: float_pre:
    :return: rgba
    see RGB_to_RGBA_and_BGR_test()
    """
    rgba = tuple([round(x / 255, fp) for x in rgb])
    rgba += (opacity,)
    return rgba


def colors_plot(ack: bool = True, tabs: int = 1, max_time: float = None):
    """
    # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    :return:
    see colors_plot_test()
    """
    n = len(COLORS_STRINGS)
    n_cols = 4
    n_rows = n // n_cols

    fig, ax = plt.subplots(figsize=(6, 5))
    set_window_title(fig, 'ColorMap({})'.format(n))
    move_figure_by_str(fig, Location.TOP_LEFT.value)
    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (n_rows + 1)
    w = X / n_cols
    print_d = {}
    for i, name in enumerate(COLORS_STRINGS):
        row = i % n_rows
        col = i // n_rows
        y = Y - (row * h) - h
        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        name_p = name
        if name in mt.CONST_COLOR_SHORTCUTS.values():
            shortcut = mt.get_key_by_value(mt.CONST_COLOR_SHORTCUTS, name)
            name_p = '{}({})'.format(name, shortcut)
        print_d[str(i)] = '{}, RGB {}, BGR {}'.format(name_p, get_rgb_color(name), get_bgr_color(name))
        ax.text(xi_text, y, name_p, fontsize=(h * 0.6), horizontalalignment='left', verticalalignment='center')
        ax.hlines(y + h * 0.1, xi_line, xf_line, color=get_rgba_color(name), linewidth=(h * 0.8))
    if ack:
        mt.dict_as_table(print_d, title='colors', tabs=tabs)
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    finalize_plot(fig, max_time)
    return


def get_ticks_list(x_low: [float, int], x_high: [float, int], p: float = 0.1) -> list:
    """
    :param x_low:
    :param x_high:
    :param p:
    :return:
    calculates a list that starts from x_low to x_high each p%
    see get_ticks_list_test()
    """
    dist = (x_high - x_low)
    p_percent_jump = dist * p
    x_ticks = [x_low + i * p_percent_jump for i in range(math.ceil(1 / p) + 1)]
    return x_ticks


def get_random_rgba_color_map(n: int, opacity: float = 1.0) -> np.array:
    """
    get colors list uniform distribution
    :param n: how many colors
    :param opacity:
    :return: np array of size(n,4). each row in RGBA format
    see get_random_RGBA_color_map_test()
    """
    colors_map = mt.np_uniform(shape=(n, 4), lows=0, highs=1)
    colors_map[:, -1] = opacity
    return colors_map


def get_random_color(c_type: str = 'str') -> (str, tuple):
    """
    random marker
    :return:
    """
    color = COLORS_STRINGS[np.random.randint(low=0, high=len(COLORS_STRINGS))]
    if c_type == 'str':
        pass
    elif c_type == 'RGB':
        color = get_rgb_color(color)
    elif c_type == 'RGBA':
        color = get_rgba_color(color)
    elif c_type == 'BGR':
        color = get_bgr_color(color)
    return color


def get_random_marker() -> (str, int):
    """
    random marker
    :return:
    """
    return MARKERS[np.random.randint(low=0, high=len(MARKERS))]


def screen_dims() -> (int, int):
    """
    :requires pip install PIL
    if your display setting as set to other 100% this function
    this function seems to overcome this on windows
    might not work
    see screen_dims_test()
    """
    try:
        if sys.platform == 'win32':
            """ this is a fix to pass windows setting display different than 100% """
            import ctypes
            user32 = ctypes.windll.user32
            window_w, window_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            # noinspection PyPackageRequirements
            from PIL import ImageGrab
            img = ImageGrab.grab()
            window_w, window_h = img.size
    except (ValueError, Exception, ModuleNotFoundError) as e:
        window_w, window_h = -1, -1
        mt.exception_error(e)
    return window_w, window_h


def move_figure_x_y(fig: matplotlib.figure, x_y: tuple) -> None:
    """
    :param fig: figure to be moved
    :param x_y: tuple of ints. x,y of top left corner
    Move figure's upper left corner to pixel (x, y)
    see move_figure_x_y_test()
    """
    try:
        x, y = x_y
        x, y = int(x), int(y)
        new_geom = "+{}+{}".format(x, y)
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            manager = fig.canvas.manager
            manager.window.wm_geometry(new_geom)
        elif backend == 'WXAgg':
            fig.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            fig.canvas.manager.window.move(x, y)
    except (ValueError, Exception) as e:
        mt.exception_error(e)
    return


def calc_x_y_by_loc_str(
        where: str,
        window_w: int, window_h: int,
        fig_w: int, fig_h: int,
        task_bar_offset: int = 0
) -> (int, int):
    x, y = 0, 0  # Location.TOP_LEFT.value: default
    if where == Location.TOP_CENTER.value:
        x = (window_w - fig_w) / 2
        y = 0
    elif where == Location.TOP_RIGHT.value:
        x = window_w - fig_w
        y = 0
    elif where == Location.CENTER_LEFT.value:
        x = 0
        y = (window_h - fig_h - task_bar_offset) / 2
    elif where == Location.CENTER_CENTER.value:
        x = (window_w - fig_w) / 2
        y = (window_h - fig_h - task_bar_offset) / 2
    elif where == Location.CENTER_RIGHT.value:
        x = window_w - fig_w
        y = (window_h - fig_h - task_bar_offset) / 2
    elif where == Location.BOTTOM_LEFT.value:
        x = 0
        y = window_h - fig_h - task_bar_offset
    elif where == Location.BOTTOM_CENTER.value:
        x = (window_w - fig_w) / 2
        y = window_h - fig_h - task_bar_offset
    elif where == Location.BOTTOM_RIGHT.value:
        x = window_w - fig_w
        y = window_h - fig_h - task_bar_offset
    x, y = int(x), int(y)
    return x, y


def move_figure_by_str(fig: matplotlib.figure, where: str = Location.TOP_LEFT.value,
                       task_bar_offset: int = None) -> None:
    """
    :param fig: figure to be moved
    :param where: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
    :param task_bar_offset: size of taskbar for bottom locs
    see move_figure_by_str_test()
    """
    where_full = Location.where_to(w_short=where)
    if where_full is not None:  # if not None, shortcut was entered
        where = where_full
    try:
        window_w, window_h = screen_dims()  # screen dims in pixels
        if window_w != -1 and window_h != -1:
            fig_w, fig_h = fig.get_size_inches() * fig.dpi  # fig dims in pixels
            if task_bar_offset is None:
                # no toolbar 75 is good. with toolbar 100
                task_bar_offset = 75 if matplotlib.rcParams['toolbar'] == 'none' else 100
            x, y = calc_x_y_by_loc_str(where, window_w, window_h, fig_w, fig_h, task_bar_offset=task_bar_offset)
            move_figure_x_y(fig=fig, x_y=(x, y))
        else:
            move_figure_x_y(fig=fig, x_y=(0, 0))
    except (ValueError, Exception) as e:
        mt.exception_error(e)
        move_figure_x_y(fig=fig, x_y=(0, 0))
    return


def set_window_title(fig: matplotlib.figure, title: str) -> None:
    """ sets window title """
    # fig.canvas.set_window_title(title)  # deprecated
    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass
    return


def set_figure_title(fig: matplotlib.figure, title: str) -> None:
    """ sets figure title """
    fig.suptitle(title)
    return


def set_axes_title(ax: matplotlib.axes, title: str):
    """ sets plot title """
    ax.set_title(title)
    return


def render_plot(fig: matplotlib.figure, block: bool = False, pause: float = 0.001) -> None:
    """
    :param fig:
    :param block:
    :param pause:
    :return:
    renders the 2d and 3d plots
    """
    # noinspection PyBroadException
    try:
        fig.canvas.draw()
        fig.show()

        if block:
            block_label = 'click to continue'
            # noinspection PyProtectedMember
            old_title_obj = fig._suptitle
            if old_title_obj is not None:
                old_title = old_title_obj.get_text()
                new_title = '{} - {}'.format(old_title, block_label)
            else:
                old_title = None
                new_title = block_label
            set_figure_title(fig, new_title)

            plt.waitforbuttonpress()
            set_figure_title(fig, old_title)
        else:
            plt.pause(pause)
    except Exception as e:  # noqa
        mt.exception_error('The plot was closed ! can\'t continue. Error: {}'.format(sys.exc_info()[0]), tabs=0)
    return


def block_x_button(event: CloseEvent) -> None:
    """
    TODO FUTURE
    block CloseEvent
    """
    print('X was clicked')
    print('\t{}'.format(event.name))
    print('\t{}'.format(event.canvas))
    print('\t{}'.format(event.guiEvent))
    # plt.gcf().stop_event_loop()
    # plt.get_current_fig_manager().stop_event_loop()
    # plt.gcf().flush_events()
    # plt.get_current_fig_manager().flush_events()
    return


def on_click_close_all(event: KeyEvent) -> None:
    """
    connects to figures and close all upon escape pressed
    :param event:
    :return:
    """
    sys.stdout.flush()
    if event.key == 'escape':
        plt.close('all')
    return


def close_all_g() -> None:
    plt.close('all')
    # plt.close()
    return


def save_plot(path: str, ack: bool = True, tabs: int = 1) -> None:
    plt.savefig(path, dpi=200, bbox_inches='tight')
    if ack:
        path = '{}.png'.format(path)
        size_s = mt.file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_s)
        print('{}{}'.format(tabs * '\t', mt.SAVED.format(file_msg)))
    return


def add_timer(fig: matplotlib.figure, max_time_seconds: float) -> any:
    """
    add timer for current fig window
    :param fig:
    :param max_time_seconds:
    :return:
    """
    # creating a timer object and setting an interval of 3000 milliseconds
    ms = max_time_seconds * 1000
    timer = fig.canvas.new_timer(interval=ms)
    timer.add_callback(close_all_g)
    timer.start()
    return timer


def finalize_plot(fig: matplotlib.figure, max_time_seconds: float = None,
                  end_str: (str, None) = 'click Esc to close') -> None:
    """
    :param fig:
    :param max_time_seconds: if not None - close plot after max_time_seconds
    :param end_str: if not None - write end_str at the end of the title
    blocks the plot
    """
    if end_str is not None:
        # noinspection PyProtectedMember
        old_title_obj = fig._suptitle
        if old_title_obj is not None:
            set_figure_title(fig, '{} - {}'.format(old_title_obj.get_text(), end_str))
        else:
            set_figure_title(fig, end_str)

    fig.canvas.mpl_connect('key_press_event', on_click_close_all)  # close on ESC pressed
    timer = None
    if max_time_seconds is not None:
        timer = add_timer(fig, max_time_seconds)
    plt.show(block=True)

    if timer is not None:  # need to stop timer
        timer.stop()
    # fig.canvas.mpl_disconnect(cid)
    return


# 2d plots
def plot_2d_many_figures_iterative_init(
        fig_d: dict,
        data_all_plots: list,
        win_d: dict = None
) -> (matplotlib.figure, list, list):
    """
    :param fig_d: dict
        'grid_xy': mandatory: tuple of ints: (rows, cols). rows*cols = number of sub plots
        'render': mandatory: bool: if to render now
        'block': optional - default False: bool: if render==True, whether to block or not
        'title': optional: str: figure title
    :param data_all_plots: mandatory: list of dicts. |data_all_plots| = rows*cols
                each dict data_plot_i has:
                    'datum': mandatory: list of data for the sub plot
                        data_i(dict): has:
                            'data': mandatory: np array or list of nx2 data points
                            'c': str: optional: color. if None - plt chooses color
                            'label': str: optional: data label for the legend
                            'marker': str: optional: marker for this data. if None - plt chooses marker
                            'marker_s': int: optional: marker size for this data. if None - plt chooses marker_s
                    'title': str: optional: title of the sub plot
                    'legend': dict: optional: legend of the subplot
                    'fig_lims': list: optional: limits of x and y axes. list of 4 ints: x_left, x_right, y_bottom, y_top
                        if you know the limits or want to avoid the limits derived from first iteration data - use this
                    'xticks': optional: list: xticks list
                    'yticks': optional: list: yticks list
    :param win_d: dict, optional(default see WINDOW_DEFAULT)
            if not None, has:
                'title': optional: str: window title
                'location': optional: str, tuple of ints or None: top left corner location
                    if tuple: x,y coordinates
                    if str: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
                'resize': optional: float>0: fig_size*=resize
                'zoomed': optional: bool: full screen or fig size
    :return:
    see plot_2d_many_figures_iterative_test()
    """

    plt.close('all')

    if win_d is None:
        win_d = WINDOW_DEFAULT

    resize = win_d['resize'] if 'resize' in win_d else None
    figsize = (6.4, 4.8) if resize is None else (6.4 * resize, 4.8 * resize)  # default figsize=(6.4, 4.8)

    rows, cols = fig_d['grid_xy']
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False, figsize=figsize)

    if 'title' in win_d and win_d['title'] is not None:
        # noinspection PyTypeChecker
        set_window_title(fig, title=win_d['title'])

    loc = win_d['location'] if 'location' in win_d and win_d['location'] is not None else Location.TOP_LEFT.value
    if mt.is_str(loc):
        move_figure_by_str(fig, where=loc)
    elif mt.is_tuple(loc):
        move_figure_x_y(fig, x_y=loc)

    if 'zoomed' in win_d and win_d['zoomed']:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')

    if 'title' in fig_d and fig_d['title'] is not None:
        set_figure_title(fig, fig_d['title'])

    axes_list = [axes] if (rows == 1 and cols == 1) else axes.flatten().tolist()

    scatters = []
    for i, ax in enumerate(axes_list):
        scatters_plot_i = []
        data_plot_i = data_all_plots[i]
        if 'title' in data_plot_i and data_plot_i['title'] is not None:
            set_axes_title(ax, data_plot_i['title'])

        saw_label = False
        for data in data_plot_i['datum']:
            c = data['c'] if 'c' in data else None
            marker = data['marker'] if 'marker' in data else None
            marker_s = data['marker_s'] if 'marker_s' in data else None
            label = data['label'] if 'label' in data else None
            if label is not None:
                saw_label = True
            sc = add_2d_scatter(ax=ax, data=data['data'], c=c, marker=marker, marker_s=marker_s, label=label)
            scatters_plot_i.append(sc)
        if saw_label and 'legend' in data_plot_i and data_plot_i['legend'] is not None:
            legend = data_plot_i['legend']
            ax.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
                      framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])
        if 'fig_lims' in data_plot_i and data_plot_i['fig_lims'] is not None:
            x_y_lims = data_plot_i['fig_lims']
            ax.set_xlim(left=x_y_lims[0], right=x_y_lims[1])
            ax.set_ylim(bottom=x_y_lims[2], top=x_y_lims[3])

        if 'xticks' in data_plot_i and data_plot_i['xticks'] is not None:
            ax.set_xticks(data_plot_i['xticks'])
        if 'yticks' in data_plot_i and data_plot_i['yticks'] is not None:
            ax.set_yticks(data_plot_i['yticks'])

        scatters.append(scatters_plot_i)

    if fig_d['render']:
        render_plot(fig=fig, block=(fig_d['block'] if '' in fig_d else False))
    return fig, axes_list, scatters


def add_2d_scatter(ax: matplotlib.axes,
                   data: (np.array, list),
                   c: (str, tuple, list),
                   marker: str,
                   marker_s: int,
                   label: str,
                   ) -> PathCollection:
    """
    :param ax:
    :param data: nx2 data. x,y rows
    :param c:
        str: e.g. 'r' ...
        tuple: RGBA color
        str, tuple or list of len 1: one color for all points 'g', (0.0, 0.5, 0.0, 1.0), ['g'], [(0.0, 0.5, 0.0, 1.0)]
        list of len n: color for each point. colors are in RGBA or str format
    :param marker: see MARKERS dict for options
    :param marker_s:
    :param label:
    :return:
    easy access to add 2d scatter
    see add_update_2d_scatter_test()
    """
    if mt.is_list(data):
        data = np.array(data)
    X, Y = mt.de_augment_numpy(data)
    sc = ax.scatter(X, Y, color=c, marker=marker, s=marker_s, label=label)
    return sc


# noinspection PyProtectedMember
def update_2d_scatter(sc: PathCollection, sc_dict: dict) -> None:
    """
    :param sc: scatter 2d pointer
    :param sc_dict: dict with data and c
    data must exist - could be none or empty array/list.
        else nx2 array of x,y points
    c optional - could be none or empty array/list.
        don't exist, None or empty - colors for this scatter stay as they were
        list of len 1: one color for all points
        list of len n: color for each point
        colors are in RGBA format
    :return:
    see add_update_2d_scatter_test()
    """
    if sc_dict['data'] is not None and len(sc_dict['data']) > 0:
        sc._offsets = sc_dict['data']
        if 'c' in sc_dict and sc_dict['c'] is not None and len(sc_dict['c']) > 0:
            sc._facecolors = sc_dict['c']
            sc._edgecolors = sc_dict['c']
    else:  # if no data
        sc._offsets = np.zeros(shape=(1, 2), dtype=np.int32)  # set a 2d point: 0,0
        if len(sc._facecolors) > 1:  # if there were more than 1 color
            sc._facecolors = [sc._facecolors[0]]  # set first color
        if len(sc._edgecolors) > 1:  # if there were more than 1 color
            sc._edgecolors = [sc._edgecolors[0]]  # set first color
    return


def plot_2d_many_figures_iterative_update(
        fig_d: dict,
        new_data_all_plots: list
) -> None:
    """
    :param fig_d: dict. has:
            'fig': matplotlib.figure: the figure
            'scatters': list of scatters for all plots
                each item is scatters_plot_i, a list of scatters in each sub plot
                scatter is PathCollection object
            'new_title': str or None: change the new title. e.g. iteration number
            'render': bool: if to render now - you can render later on plot_2d_iterative_plot_update()
            'block': bool: if render==True, whether to block or not
    :param new_data_all_plots: list. |new_data_all_plots|==|fig_d['scatters']|
            each item is new_data_plot_i: list. |new_data_plot_i|==|scatters_plot_i|
                each item is new_data_i: dict with:
                    'data': nx2 data points - could be none or empty - no refresh of data
                    'c': None, tuple or list:
                        None - keep colors
                        tuple - RGBA color. change all points to this color
                        list - RGBA colors of size n. change point i to color i

    see plot_2d_many_figures_iterative_test()
    """
    if 'new_title' in fig_d and fig_d['new_title'] is not None:
        set_figure_title(fig_d['fig'], fig_d['new_title'])

    for i, scatters_plot_i in enumerate(fig_d['scatters']):  # for each sub plot
        new_data_plot_i = new_data_all_plots[i]
        for j, scatter in enumerate(scatters_plot_i):  # for each scatter in sub plot
            update_2d_scatter(sc=scatter, sc_dict=new_data_plot_i[j])

    if fig_d['render']:
        render_plot(fig=fig_d['fig'], block=fig_d['block'])
    return


def plot_2d_one_figure_iterative_init(
        fig_d: dict,
        data_plot_i: dict,
        win_d: dict = None
) -> (matplotlib.figure, list, list):
    """
    :param fig_d: dict
        'render': mandatory: bool: if to render now
        'block': optional - default False: bool: if render==True, whether to block or not
        'title': optional: str: figure title
    :param data_plot_i: mandatory: dict.
                data_plot_i has:
                    'datum': mandatory: list of data for the sub plot
                        data_i(dict): has:
                            'data': mandatory: np array or list of nx2 data points
                            'c': str: optional: color. if None - plt chooses color
                            'label': str: optional: data label for the legend
                            'marker': str: optional: marker for this data. if None - plt chooses marker
                            'marker_s': int: optional: marker size for this data. if None - plt chooses marker_s
                    'title': str: optional: title of the sub plot
                    'legend': dict: optional: legend of the subplot
                    'fig_lims': list: optional: limits of x and y axes. list of 4 ints: x_left, x_right, y_bottom, y_top
                        if you know the limits or want to avoid the limits derived from first iteration data - use this
                    'xticks': optional: list: xticks list
                    'yticks': optional: list: yticks list
    :param win_d: dict, optional(default see WINDOW_DEFAULT)
            if not None, has:
                'title': optional: str: window title
                'location': optional: str, tuple of ints or None: top left corner location
                    if tuple: x,y coordinates
                    if str: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
                'resize': optional: float>0: fig_size*=resize
                'zoomed': optional: bool: full screen or fig size
    :return:
    see plot_2d_one_figure_iterative_test()
    """

    fig, axes, scatters = plot_2d_many_figures_iterative_init(
        fig_d={
            'title': fig_d['title'],
            'grid_xy': (1, 1),
            'render': fig_d['render'],
            'block': fig_d['block'],
        },
        data_all_plots=[data_plot_i],
        win_d=win_d,
    )
    return fig, axes, scatters[0]


def plot_2d_one_figure_iterative_update(
        fig_d: dict,
        new_data_plot_i: list
) -> None:
    """
    :param fig_d: dict. has:
            'fig': matplotlib.figure: the figure
            'scatters': scatters_plot_i, a list of scatters in the plot
                scatter is PathCollection object
            'new_title': str or None: change the new title. e.g. iteration number
            'render': bool: if to render now - you can render later on plot_2d_iterative_plot_update()
            'block': bool: if render==True, whether to block or not
    :param new_data_plot_i: list
                each item is new_data_i: dict with:
                    'data': nx2 data points - could be none or empty - no refresh of data
                    'c': None, tuple or list:
                        None - keep colors
                        tuple - RGBA color. change all points to this color
                        list - RGBA colors of size n. change point i to color i

    see plot_2d_one_figure_iterative_test()
    """
    plot_2d_many_figures_iterative_update(
        fig_d={
            'fig': fig_d['fig'],
            'scatters': [fig_d['scatters']],
            'new_title': fig_d['new_title'],
            'render': fig_d['render'],
            'block': fig_d['block'],
        },
        new_data_all_plots=[new_data_plot_i],
    )
    return


def plot_2d_iterative_dashboards_init(
        fig_d: dict,
        data_all_plots: list,
        win_d: dict = None,
        center_d: dict = None
) -> (matplotlib.figure, list, list):
    """
    iterative functions: difference between dashboard and normal figure is:
        in dashboard you know the size of the data coming (x,y lims)
        in dashboard no data on build up
    :param fig_d: dict
        'grid_xy': mandatory: tuple of ints: (rows, cols). rows*cols = number of sub plots
        'render': mandatory: bool: if to render now
        'block': optional - default False: bool: if render==True, whether to block or not
        'title': optional: str: figure title
    :param data_all_plots: mandatory: list of dicts. |data_all_plots| = rows*cols
        each dict data_plot_i has:
            'fig_lims': list: mandatory: limits of x and y axes. list of 4 ints: x_left, x_right, y_bottom, y_top
            'datum': mandatory: list of data for the sub plot
                data_i(dict): has:
                    'c': str: optional: color. if None - plt chooses color
                    'label': str: optional: data label for the legend
                    'marker': str: optional: marker for this data. if None - plt chooses marker
                    'marker_s': int: optional: marker size for this data. if None - plt chooses marker_s
            'title': str: optional: title of the sub plot
            'legend': dict: optional: legend of the subplot
            'xticks': optional: list: xticks list
            'yticks': optional: list: yticks list
    :param win_d: dict, optional(default see WINDOW_DEFAULT)
        if not None, has:
            'title': optional: str: window title
            'location': optional: str, tuple of ints or None: top left corner location
                if tuple: x,y coordinates
                if str: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
            'resize': optional: float>0: fig_size*=resize
            'zoomed': optional: bool: full screen or fig size
    :param center_d: dict or None. draw figure center if not None
        e.g. CENTER_DEFAULT
    :return:
    see plot_2d_iterative_dashboard_test()
        """
    plt.close('all')

    if win_d is None:
        win_d = WINDOW_DEFAULT

    resize = win_d['resize'] if 'resize' in win_d else None
    figsize = (6.4, 4.8) if resize is None else (6.4 * resize, 4.8 * resize)  # default figsize=(6.4, 4.8)

    rows, cols = fig_d['grid_xy']
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False, figsize=figsize)

    if 'title' in win_d and win_d['title'] is not None:
        # noinspection PyTypeChecker
        set_window_title(fig, win_d['title'])
    loc = win_d['location'] if 'location' in win_d and win_d['location'] is not None else Location.TOP_LEFT.value
    if mt.is_str(loc):
        move_figure_by_str(fig, where=loc)
    elif mt.is_tuple(loc):
        move_figure_x_y(fig, x_y=loc)

    if 'zoomed' in win_d and win_d['zoomed']:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')

    if 'title' in fig_d and fig_d['title'] is not None:
        set_figure_title(fig, fig_d['title'])

    axes_list = [axes] if (rows == 1 and cols == 1) else axes.flatten().tolist()

    scatters = []
    for i, ax in enumerate(axes_list):
        ax.set_aspect('equal', adjustable='box')
        data_plot_i = data_all_plots[i]
        if 'title' in data_plot_i and data_plot_i['title'] is not None:
            set_axes_title(ax, data_plot_i['title'])
        scatters_plot_i = []
        saw_label = False
        for data in data_plot_i['datum']:
            c = data['c'] if 'c' in data else None
            marker = data['marker'] if 'marker' in data else None
            marker_s = data['marker_s'] if 'marker_s' in data else None
            label = data['label'] if 'label' in data else None
            if label is not None:
                saw_label = True
            origin = np.zeros(2)
            sc = add_2d_scatter(ax=ax, data=origin, c=c, marker=marker, marker_s=marker_s, label=label)
            scatters_plot_i.append(sc)

        x_y_lims = data_plot_i['fig_lims']
        ax.set_xlim(left=x_y_lims[0], right=x_y_lims[1])
        ax.set_ylim(bottom=x_y_lims[2], top=x_y_lims[3])

        if 'xticks' in data_plot_i and data_plot_i['xticks'] is not None:
            ax.set_xticks(data_plot_i['xticks'])
        if 'yticks' in data_plot_i and data_plot_i['yticks'] is not None:
            ax.set_yticks(data_plot_i['yticks'])

        if center_d is not None:
            x_left, x_right = ax.get_xlim()
            y_bottom, y_top = ax.get_ylim()

            cx = int((x_right - x_left) / 2 + x_left)
            cy = int((y_top - y_bottom) / 2 + y_bottom)
            c_xy = np.array([cx, cy])
            _ = add_2d_scatter(ax=ax, data=c_xy, c=center_d['c'],
                               marker=center_d['marker'], marker_s=center_d['marker_size'],
                               label='{}({},{})'.format(center_d['label'], cx, cy))
        if saw_label and 'legend' in data_plot_i and data_plot_i['legend'] is not None:
            legend = data_plot_i['legend']
            ax.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
                      framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])
        scatters.append(scatters_plot_i)

    if fig_d['render']:
        render_plot(fig=fig, block=(fig_d['block'] if '' in fig_d else False))
    return fig, axes_list, scatters


def plot_2d_iterative_dashboard_init(
        fig_d: dict,
        data_plot_i: dict,
        win_d: dict = None,
        center_d: dict = None
) -> (matplotlib.figure, list, list):
    """
    iterative functions: difference between dashboard and normal figure is:
        in dashboard you know the size of the data coming (x,y lims)
        in dashboard no data on build up
    :param fig_d: dict
        'render': mandatory: bool: if to render now
        'block': optional - default False: bool: if render==True, whether to block or not
        'title': optional: str: figure title
    :param data_plot_i: mandatory: dict.
        data_plot_i has:
            'fig_lims': list: mandatory: limits of x and y axes. list of 4 ints: x_left, x_right, y_bottom, y_top
            'datum': mandatory: list of data for the sub plot
                data_i(dict): has:
                    'c': str: optional: color. if None - plt chooses color
                    'label': str: optional: data label for the legend
                    'marker': str: optional: marker for this data. if None - plt chooses marker
                    'marker_s': int: optional: marker size for this data. if None - plt chooses marker_s
            'title': str: optional: title of the sub plot
            'legend': dict: optional: legend of the subplot
            'xticks': optional: list: xticks list
            'yticks': optional: list: yticks list
    :param win_d: dict, optional(default see WINDOW_DEFAULT)
        if not None, has:
            'title': optional: str: window title
            'location': optional: str, tuple of ints or None: top left corner location
                if tuple: x,y coordinates
                if str: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
            'resize': optional: float>0: fig_size*=resize
            'zoomed': optional: bool: full screen or fig size
    :param center_d: dict or None. draw figure center if not None
        e.g. CENTER_DEFAULT
    :return:
    see plot_2d_iterative_dashboard_test()
    """

    fig, axes, scatters = plot_2d_iterative_dashboards_init(
        fig_d={
            'title': fig_d['title'],
            'grid_xy': (1, 1),
            'render': fig_d['render'],
            'block': fig_d['block'],
        },
        data_all_plots=[data_plot_i],
        win_d=win_d,
        center_d=center_d,
    )
    return fig, axes, scatters[0]


def plot_2d_one_figure(
        datum: list,
        fig_title: str = None,
        win_d: dict = None,
        max_time: int = None
) -> None:
    """
    see documentation in plot_2d_iterative_plot_init()
    see plot_2d_one_figure_test()
    """

    fig, axes, scatters_plot_i = plot_2d_one_figure_iterative_init(
        fig_d={
            'title': fig_title,
            'render': False,
            'block': False,
        },
        data_plot_i={
            'title': None,
            'legend': LEGEND_DEFAULT,
            'fig_lims': None,
            'xticks': None,
            'yticks': None,
            'datum': datum,
        },
        win_d=win_d
    )

    finalize_plot(fig, max_time_seconds=max_time)
    return


def plot_2d_many_figures(
        grid: tuple,
        datum_list: list,
        sub_titles: list = None,
        fig_title: str = None,
        win_d: dict = None,
        max_time: float = None
) -> (matplotlib.figure, list, list):
    """
    see documentation in plot_2d_iterative_plots_init()
    see plot_2d_scatters_test()
    """

    sub_plots = grid[0] * grid[1]
    data_all_plots = []
    for i in range(sub_plots):
        data_plot_i = {
            'title': sub_titles[i],
            'legend': LEGEND_DEFAULT,
            'fig_lims': None,
            'xticks': None,
            'yticks': None,
            'datum': datum_list[i],
        }
        data_all_plots.append(data_plot_i)

    fig, axes, scatters_plot_i = plot_2d_many_figures_iterative_init(
        fig_d={
            'title': fig_title,
            'grid_xy': grid,
            'render': False,
            'block': False,
        },
        data_all_plots=data_all_plots,
        win_d=win_d
    )

    finalize_plot(fig, max_time)
    return


def plot_x_y_std(
        data_x: np.array,
        groups: list,
        title: str = None,
        legend: dict = None,
        x_label: str = 'Size',
        y_label: str = 'Error',
        save_path: str = None,
        tabs: int = 1,
        show_plot: bool = True,
        with_shift: bool = False,
        max_time: float = None
) -> None:
    """
    :param data_x: x values for all groups
    :param groups: list of groups s.t. each tuple(y values, y std, color, title)  y std could be None
    :param title:
    :param legend: if None-no legend, else dict.
        e.g. {'loc': 'upper right', 'ncol': 1, 'fancybox': True, 'framealpha': 0.5, 'edgecolor': 'black'}
    :param x_label:
    :param y_label:
    :param save_path:
    :param tabs:
    :param show_plot:
    :param with_shift: moves a bit the point x's so it won't be one on another - blocking the view
    :param max_time:
    see plot_x_y_std_test()
    """
    plt.close('all')
    data_x_last = data_x  # in order to see all STDs, move a little on the x axis
    data_x_jump = 0.5
    data_x_offset = - int(len(groups) / 2) * data_x_jump
    line_style = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 2, "elinewidth": 1, "capsize": 4}
    for i, group in enumerate(groups):
        data_y, std_y = group[0], group[1]  # std_y could be None
        color, label = group[2], group[3]
        if with_shift:  # move x data for each set a bit so you can see it clearly
            dx_shift = [x + i * data_x_jump + data_x_offset for x in data_x]
            data_x_last = dx_shift
        plt.errorbar(data_x_last, data_y, std_y, color=color, fmt='.', label=label, **line_style)

    plt.grid()
    if legend is not None:
        plt.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
                   framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])

    if title is not None:
        set_figure_title(plt.gcf(), title)
    plt.xticks(data_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print('{}Saved: {}.png'.format(tabs * '\t', save_path))
    if show_plot:
        move_figure_by_str(plt.gcf())
        plt.pause(0.001)
        finalize_plot(plt.gcf(), max_time)
    return


def histogram(values: np.array, title: str, bins_n: int = 50, save_path: str = None, tabs: int = 1,
              max_time: float = None) -> None:
    """
    :param values:
    :param title:
    :param bins_n:
    :param save_path:
    :param tabs:
    :param max_time:
    plots a histogram
    see histogram_test()
    """
    plt.close('all')
    plt.hist(values, bins_n, density=False, facecolor='blue', alpha=0.75)
    plt.xlabel('Values')
    plt.ylabel('Bin Count')
    set_figure_title(plt.gcf(), title)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print('{}Saved: {}.png'.format(tabs * '\t', save_path))
    move_figure_by_str(plt.gcf())
    finalize_plot(plt.gcf(), max_time)
    return


def compare_images_sets(set_a, set_b, title: str = None, max_time: float = None) -> None:
    """
    build for images BEFORE transform:
    notice images should be in the format:
        gray scale mnist: [number of images, 28, 28]
        RGB  Cifar10    : [number of images, 32, 32, 3]

    :param set_a: array (nd\torch) of images
    :param set_b: array (nd\torch) of images
    :param title: plot title
    :param max_time:
    plot set a of images in row 1 and set b in row 2
    set_a and set_b can be ndarray or torch arrays
    see compare_images_sets_test()
    """
    plt.close('all')
    n_cols = max(set_a.shape[0], set_b.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, sharex='all', sharey='all', figsize=(15, 4))
    for images, row in zip([set_a, set_b], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if title is not None:
        set_figure_title(plt.gcf(), title)
    move_figure_by_str(plt.gcf())
    finalize_plot(plt.gcf(), max_time)
    return


def compare_images_multi_sets_squeezed(sets_dict: dict, title: str = None, ack: bool = True, tabs: int = 0,
                                       max_time: float = None) -> None:
    """
    build for images AFTER transform:
    notice images should be in the format:
        gray scale mnist: [number of images, 1, 28, 28]
        RGB  Cifar10    : [number of images, 3, 32, 32]

    :param sets_dict: each entry in dict is title, set of images(np/tensor) - unlimited amount allowed
    :param title: for plot
    :param ack: show str with details which set in each row
    :param tabs:
    :param max_time:
    plot sets of images in rows
    see compare_images_multi_sets_squeezed_test()
    """
    try:
        from wizzi_utils.torch import torch_tools as tt
        # noinspection PyPackageRequirements
        from torchvision.utils import make_grid
        # noinspection PyPackageRequirements
        import torch

        plt.close('all')
        for k, v in sets_dict.items():
            if mt.is_numpy(sets_dict[k]):
                sets_dict[k] = tt.numpy_to_torch(sets_dict[k])

        all_sets = None
        msg = ''
        set_len = 0
        msg_base = 'row {}: {}, '

        for i, (k, v) in enumerate(sets_dict.items()):
            all_sets = v if all_sets is None else torch.cat((all_sets, v), 0)
            msg += msg_base.format(i, k)
            set_len = v.shape[0]

        grid_images = make_grid(all_sets, nrow=set_len)
        if title is not None:
            plt.title(title)
        plt.axis('off')
        if ack:
            print('{}{}'.format(tabs * '\t', msg))
        plt.imshow(np.transpose(tt.torch_to_numpy(grid_images), (1, 2, 0)))
        move_figure_by_str(plt.gcf())
        finalize_plot(plt.gcf(), max_time)
    except ModuleNotFoundError as e:
        mt.exception_error(e)
    return


# 3d plots
def plot_3d_iterative_dashboard_init(
        fig_d: dict,
        data_plot_i: dict,
        win_d: dict = None,
        center_d: dict = None,
) -> (matplotlib.figure, Axes3D, list):
    """
    iterative functions: difference between dashboard and normal figure is:
        in dashboard you know the size of the data coming (x,y lims)
        in dashboard no data on build up
    :param fig_d: dict
        'render': mandatory: bool: if to render now
        'block': optional - default False: bool: if render==True, whether to block or not
        'title': optional: str: figure title
    :param data_plot_i: mandatory: dict.
        each dict data_plot_i has:
            'fig_lims': list: mandatory: limits of x,y,z axes.
                list of 6 ints: x_left, x_right, y_bottom, y_top, z_in, z_out
            'datum': mandatory: list of data for the sub plot
                data_i(dict): has:
                    'c': str or tuple: optional: color ('r' or RGBA('r)). if None - plt chooses color
                    'label': str: optional: data label for the legend
                    'marker': str: optional: marker for this data. if None - plt chooses marker
                    'marker_s': int: optional: marker size for this data. if None - default is 10
            'title': str: optional: title of the sub plot
            'legend': dict: optional: legend of the subplot
            'color_axes': str: optional: color of the ticks and axes labels
            'xticks': optional: list: xticks list
            'yticks': optional: list: yticks list
            'view': optional: dict - view onto the world. e.g. {'azim': 90.0, 'elev': -100.0}
            'face_color': str: optional: color of the whole background - default is white
            'background_color': str: color of the whole background - default is white
    :param win_d: dict, optional(default see WINDOW_DEFAULT)
        if not None, has:
            'title': optional: str: window title
            'location': optional: str, tuple of ints or None: top left corner location
                if tuple: x,y coordinates
                if str: see Location enum. e.g. Location.TOP_LEFT.value (which is 'top_left')
            'resize': optional: float>0: fig_size*=resize
            'zoomed': optional: bool: full screen or fig size
    :param center_d: dict or None. draw figure center if not None
        e.g. CENTER_DEFAULT
    :return:

    see plot_3d_iterative_dashboard_test()
    """
    plt.close('all')
    if win_d is None:
        win_d = WINDOW_DEFAULT

    resize = win_d['resize'] if 'resize' in win_d else None
    figsize = (6.4, 4.8) if resize is None else (6.4 * resize, 4.8 * resize)  # default figsize=(6.4, 4.8)
    fig = plt.figure(figsize=figsize)

    if 'title' in win_d and win_d['title'] is not None:
        # noinspection PyTypeChecker
        set_window_title(fig, win_d['title'])
    if 'title' in fig_d and fig_d['title'] is not None:
        set_figure_title(fig, fig_d['title'])

    loc = win_d['location'] if 'location' in win_d and win_d['location'] is not None else Location.TOP_LEFT.value
    if mt.is_str(loc):
        move_figure_by_str(fig, where=loc)
    elif mt.is_tuple(loc):
        move_figure_x_y(fig, x_y=loc)

    if 'zoomed' in win_d and win_d['zoomed']:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')

    ax = Axes3D(fig)

    # if 'title' in data_plot_i and data_plot_i['title'] is not None:  # TODO fix
    #     set_axes_title(ax, data_plot_i['title'])

    if 'face_color' in data_plot_i and data_plot_i['face_color'] is not None:
        ax.set_facecolor(data_plot_i['face_color'])

    if 'background_color' in data_plot_i and data_plot_i['background_color'] is not None:
        ax_background = get_rgba_color(data_plot_i['background_color'])
        ax.w_xaxis.set_pane_color(ax_background)
        ax.w_yaxis.set_pane_color(ax_background)
        ax.w_zaxis.set_pane_color(ax_background)

    x_y_z_lims = data_plot_i['fig_lims']
    ax.set_xlim3d(left=x_y_z_lims[0], right=x_y_z_lims[1])
    ax.set_ylim3d(bottom=x_y_z_lims[2], top=x_y_z_lims[3])
    ax.set_zlim3d(bottom=x_y_z_lims[4], top=x_y_z_lims[5])

    if 'xticks' in data_plot_i and data_plot_i['xticks'] is not None:
        ax.set_xticks(data_plot_i['xticks'])
    if 'yticks' in data_plot_i and data_plot_i['yticks'] is not None:
        ax.set_yticks(data_plot_i['yticks'])
    if 'zticks' in data_plot_i and data_plot_i['zticks'] is not None:
        ax.set_zticks(data_plot_i['zticks'])

    if 'color_axes' in data_plot_i and data_plot_i['color_axes'] is not None:
        c = data_plot_i['color_axes']
        ax.w_xaxis.line.set_color(c)
        ax.w_yaxis.line.set_color(c)
        ax.w_zaxis.line.set_color(c)
        ax.tick_params(axis='x', colors=c)
        ax.tick_params(axis='y', colors=c)
        ax.tick_params(axis='z', colors=c)
        ax.set_xlabel("X", color=c)
        ax.set_ylabel("Y", color=c)
        ax.set_zlabel("Z", color=c)
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    saw_label = False
    scatters_plot_i = []
    for data in data_plot_i['datum']:
        c = data['c'] if 'c' in data else None
        marker = data['marker'] if 'marker' in data else None
        marker_s = data['marker_s'] if 'marker_s' in data else 10
        label = data['label'] if 'label' in data else None
        if label is not None:
            saw_label = True
        origin = np.zeros(3)
        sc = add_3d_scatter(ax=ax, data=origin, c=c, marker=marker, marker_s=marker_s, label=label)
        scatters_plot_i.append(sc)

    if center_d is not None:
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        z_in, z_out = ax.get_zlim()

        cx = int((x_right - x_left) / 2 + x_left)
        cy = int((y_top - y_bottom) / 2 + y_bottom)
        cz = int((z_out - z_in) / 2 + z_in)
        c_xyz = np.array([cx, cy, cz])
        _ = add_3d_scatter(ax=ax, data=c_xyz, c=center_d['c'], marker=center_d['marker'],
                           marker_s=center_d['marker_size'],
                           label='{}({},{},{})'.format(center_d['label'], cx, cy, cz))

    if saw_label and 'legend' in data_plot_i and data_plot_i['legend'] is not None:
        legend = data_plot_i['legend']
        ax.legend(loc=legend['loc'], ncol=legend['ncol'], fancybox=legend['fancybox'],
                  framealpha=legend['framealpha'], edgecolor=legend['edgecolor'])

    if 'view' in data_plot_i and data_plot_i['view'] is not None:
        view = data_plot_i['view']
        ax.view_init(azim=view['azim'], elev=view['elev'])

    if fig_d['render']:
        render_plot(fig=fig, block=(fig_d['block'] if 'block' in fig_d else False))
    return fig, ax, scatters_plot_i


def add_3d_scatter(ax: Axes3D,
                   data: (np.array, list),
                   c: (str, tuple, list),
                   marker: str,
                   marker_s: int,
                   label: str,
                   ) -> Path3DCollection:
    """
    :param ax:
    :param data: nx3 data
    :param c:
        str: e.g. 'r' ...
        tuple: RGBA color
        str, tuple or list of len 1: one color for all points 'g', (0.0, 0.5, 0.0, 1.0), ['g'], [(0.0, 0.5, 0.0, 1.0)]
        list of len n: color for each point. colors are in RGBA or str format
    :param marker: see MARKERS dict for options
    :param marker_s:
    :param label:
    :return:
    easy access to add 3d scatter
    see add_update_3d_scatter_test()
    """
    if mt.is_list(data):
        data = np.array(data)
    X_Y, Z = mt.de_augment_numpy(data)
    X, Y = mt.de_augment_numpy(X_Y)
    sc = ax.scatter(X, Y, Z, color=c, marker=marker, s=marker_s, label=label)
    return sc


# noinspection PyProtectedMember
def update_3d_scatter(sc: Path3DCollection, sc_dict: dict) -> None:
    """
    :param sc: scatter 3d pointer
    :param sc_dict: dict with data and c
    data must exist - could be none or empty array/list.
        else nx3 array/list of x,y,z points
    c optional - could be none or empty array/list.
        don't exist, None or empty - colors for this scatter stay as they were
        str, RGBA tuple or list of len 1: one color for all points
        list of len n: list of str or RGBA: color for each point
    :return:
    see add_update_3d_scatter_test()
    """
    if sc_dict['data'] is not None and len(sc_dict['data']) > 0:
        sc._offsets3d = sc_dict['data'].T
        if 'c' in sc_dict and sc_dict['c'] is not None and len(sc_dict['c']) > 0:
            sc._facecolor3d = sc_dict['c']
            sc._edgecolor3d = sc_dict['c']
    else:  # if no data
        sc._offsets3d = np.zeros(shape=(3, 1), dtype=np.int32)  # set a 3d point: 0,0,0
        if len(sc._facecolor3d) > 1:  # if there were more than 1 color
            sc._facecolor3d = [sc._facecolor3d[0]]  # set first color
        if len(sc._edgecolor3d) > 1:  # if there were more than 1 color
            sc._edgecolor3d = [sc._edgecolor3d[0]]  # set first color
    return


def plot_3d_one_figure_iterative_update(
        fig_d: dict,
        new_data_plot_i: list,
) -> None:
    """
    :param fig_d: dict. has:
            'fig': matplotlib.figure: the figure
            'scatters': scatters_plot_i, a list of scatters in the plot
                scatter is PathCollection object
            'new_title': str or None: change the new title. e.g. iteration number
            'render': bool: if to render now - you can render later on plot_2d_iterative_plot_update()
            'block': bool: if render==True, whether to block or not
    :param new_data_plot_i: list
                each item is new_data_i: dict with:
                    'data': nx2 data points - could be none or empty - no refresh of data
                    'c': None, str, tuple or list:
                        None - keep colors
                        str - string color('r'). change all points to this color
                        tuple - RGBA color. change all points to this color
                        list - RGBA or str colors of size n. change point i to color i

    see plot_3d_iterative_dashboard_test()
    """

    if 'new_title' in fig_d and fig_d['new_title'] is not None:
        set_figure_title(fig_d['fig'], fig_d['new_title'])

    for j, scatter in enumerate(fig_d['scatters']):  # for each scatter in sub plot
        update_3d_scatter(sc=scatter, sc_dict=new_data_plot_i[j])

    if fig_d['render']:
        render_plot(fig=fig_d['fig'], block=fig_d['block'])
    return


def plot_3d_one_dashboard(
        data_plot_i: dict,
        win_d: dict = None,
        center_d: dict = None,
        fig_title: str = None,
        max_time: float = None
) -> None:
    """
    iterative functions: difference between dashboard and normal figure is:
        in dashboard you know the size of the data coming (x,y lims)
        in dashboard no data on build up
    see documentation in plot_3d_iterative_dashboard_init() and plot_3d_one_figure_iterative_update()
    see plot_3d_one_dashboard()
    """
    fig, ax, iterative_scatters = plot_3d_iterative_dashboard_init(
        fig_d={
            'title': fig_title,
            'render': False,
            'block': False,
        },
        data_plot_i=data_plot_i,
        win_d=win_d,
        center_d=center_d,
    )

    plot_3d_one_figure_iterative_update(
        fig_d={
            'fig': fig,
            'scatters': iterative_scatters,
            'new_title': None,
            'render': False,
            'block': False,
        },
        new_data_plot_i=data_plot_i['datum'],
    )

    finalize_plot(fig, max_time)
    return


def plot_3d_cube(
        axes: Axes3D,
        cube_definition: list,
        label='cube',
        face_color: str = 'blue',
        face_opacity: float = 0.1,
        edge_color: str = 'black',
        edge_width: float = 1.0,
        add_corners_coordinates: bool = False,
) -> None:
    """
    :param axes:
    :param cube_definition: list of np.arrays
        4 points of the cube (1 corner node and 3 nodes to the left right and top)
    :param label:
    :param face_color:
    :param face_opacity:
    :param edge_color:
    :param edge_width:
    :param add_corners_coordinates: add cube corners label on figure
    see plot_3d_cube_test()
    """

    points = []
    points += cube_definition
    vectors = [
        cube_definition[1] - cube_definition[0],
        cube_definition[2] - cube_definition[0],
        cube_definition[3] - cube_definition[0]
    ]

    points += [cube_definition[0] + vectors[0] + vectors[1]]
    points += [cube_definition[0] + vectors[0] + vectors[2]]
    points += [cube_definition[0] + vectors[1] + vectors[2]]
    points += [cube_definition[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=edge_width, edgecolors=edge_color)
    # noinspection PyUnresolvedReferences
    color_rgba = matplotlib.colors.to_rgba(face_color, alpha=face_opacity)
    faces.set_facecolor(color_rgba)
    axes.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, color=edge_color, label=label)

    if add_corners_coordinates:
        for p in points:
            x, y, z = p
            text = '{:.2f},{:.2f},{:.2f}'.format(x, y, z)
            axes.text(x, y, z, text, zdir=(1, 1, 1))
    return


def add_cube3d_around_origin(axes: Axes3D, edge_len: int, color: str = 'b', add_labels: bool = False) -> None:
    """
    :param axes:
    :param edge_len: cube edge size
    :param color: cube color
    :param add_labels: add cube labels on the scene
    see add_cube3d_around_origin_test()
    """
    half_edge = int(edge_len / 2)
    xyz_bot_left = np.array([-half_edge, -half_edge, -half_edge], dtype=float)
    xyz_top_left = np.copy(xyz_bot_left)
    xyz_bot_right = np.copy(xyz_bot_left)
    xyz_bot_left_depth = np.copy(xyz_bot_left)
    xyz_top_left[1] += edge_len  # add just y
    xyz_bot_right[0] += edge_len  # add just x
    xyz_bot_left_depth[2] += edge_len  # add just z

    cube_4_edges = [xyz_bot_left, xyz_top_left, xyz_bot_right, xyz_bot_left_depth]
    plot_3d_cube(
        axes=axes,
        cube_definition=cube_4_edges,
        label='cube(edge={})'.format(edge_len),
        face_color=color,
        add_corners_coordinates=add_labels
    )
    return


def plt_dark_mode() -> None:
    plt.style.use('dark_background')
    return


def remove_toolbar() -> None:
    matplotlib.rcParams['toolbar'] = 'None'
    return
