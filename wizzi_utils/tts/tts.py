import sys
import os
import subprocess
import inspect
# noinspection PyPackageRequirements
import pyttsx3  # pip install pyttsx3


# @staticmethod
# def onStart(name):
#     print('starting', name)
#
# @staticmethod
# def onWord(name, location, length):
#     print('word', name, location, length)
#
# @staticmethod
# def onEnd(name, completed):
#     print('finishing', name, completed)

# self.buddy = pyttsx3.init()
# self.buddy.connect('started-utterance', self.onStart)
# self.buddy.connect('started-word', self.onWord)
# self.buddy.connect('finished-utterance', self.onEnd)

def get_file_name() -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[1]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.filename)
    except IndexError as e:
        print(e)
    return ret_val


class MachineBuddy:
    """
    text to speech
    https://pypi.org/project/pyttsx3/
    """

    def __init__(self, voice_ind: int = 0, rate: int = 200, vol: float = 1.0) -> None:
        """
        :param voice_ind: default voice - num 0 in self.buddy.getProperty('voices')
        :param rate: default rate 200
        :param vol: default vol 1.0 (from 0.0 to 1.0)
        """
        self.buddy = pyttsx3.init()
        # noinspection PyTypeChecker
        self.voices_num = len(self.buddy.getProperty('voices'))
        self.current_voice_ind = 0
        if 0 < voice_ind <= self.voices_num - 1:  # 0 by default
            self.change_voice(voice_ind)
        if 0 < rate != 200:
            self.change_rate(rate)
        if 0.0 <= vol < 1.0:
            self.change_volume(vol)
        self.buddy.runAndWait()
        return

    def __del__(self) -> None:
        # if self.buddy is not None:
        #     self.buddy.stop()
        return

    def __str__(self) -> str:
        string = 'MachineBuddy:\n'
        string += '\tvoice({}):{}\n'.format(self.get_current_property(key='voice_idx'),
                                            self.get_current_property(key='voice_name'))
        string += '\tvol:{}\n'.format(self.get_current_property(key='volume'))
        string += '\trate:{}'.format(self.get_current_property(key='rate'))
        return string

    def get_all_voices(self, ack: bool = True, tabs: int = 1):
        """
        :param ack:
        :param tabs:
        :return:
        TO add more speakers on Windows10:
        based on https://www.youtube.com/watch?v=KMtLqPi2wiU&ab_channel=MuruganS
        step 1(download):
            ctrl+winKey+n - open narrator
            add more voices
            add voice
            select, install and wait for it to finish
            # the new voice available only to Microsoft
            i will refer to this language as _NEW_LANGUAGE
        step 2(make the installed voice available to all tts programs):
            start, regedit, open as administrator:
                goto: Computer/HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens
                click export on any language and save to desktop(we just need a path) -> i will call it f1
                goto: Computer/HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE
                select the _NEW_LANGUAGE installed on step 1. export to desktop -> i will call it f2
                open both saved files on notepad++
                from f1 - copy the paths
                e.g.
                path1: [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/TTS_MS_EN-GB_HAZEL_11.0]
                path2: [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/TTS_MS_EN-GB_HAZEL_11.0/Attributes]
                we care only for the dir name:
                so path1: [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/SOME_LANGUAGE]
                so path2: [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/SOME_LANGUAGE/Attributes]

                open f2
                duplicate the first 2 paragraphs (first is the actual language and second is the attributes)
                paragraph 1:
                    [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE]
                paragraph 2:
                    [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE/Attributes]
                now you have 4 paragraphs instead of 2:
                p1 [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE]
                p2 [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE/Attributes]
                p3 [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE]
                p4 [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/_NEW_LANGUAGE/Attributes]
                change p3 to path1 and change SOME_LANGUAGE to _NEW_LANGUAGE
                change p4 to path2 and change SOME_LANGUAGE to _NEW_LANGUAGE
                e.g.
                from
                [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/MSTTS_V110_heIL_Asaf]
                to
                [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/MSTTS_V110_heIL_Asaf]
                and from
                [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech_OneCore/Voices/Tokens/MSTTS_V110_heIL_Asaf/Attributes]
                to
                [HKEY_LOCAL_MACHINE/SOFTWARE/Microsoft/Speech/Voices/Tokens/MSTTS_V110_heIL_Asaf/Attributes]
                save and exit notepad++
                run the saved file and _NEW_LANGUAGE should appear in any tts program
        """

        # all_voice = self.buddy.getProperty('voices')
        all_voices = self.get_current_property('all_voices')
        if ack:
            # noinspection PyTypeChecker
            for v in all_voices:
                print('{}{}'.format(tabs * '\t', v))
        return all_voices

    def change_voice(self, new_voice_ind: int) -> None:
        """
        :param new_voice_ind:
        voices differ between computers
        :return:
        """
        if 0 <= new_voice_ind <= self.voices_num - 1:
            all_voices = self.get_current_property('all_voices')
            # noinspection PyUnresolvedReferences
            self.buddy.setProperty(name='voice', value=all_voices[new_voice_ind].id)
            self.current_voice_ind = new_voice_ind
            self.buddy.runAndWait()
        return

    def change_rate(self, new_rate: int) -> None:
        """
        default rate 200
        :param new_rate:
        :return:
        """
        if new_rate > 0:
            self.buddy.setProperty('rate', new_rate)  # setting up new voice rate
            self.buddy.runAndWait()
        return

    def change_volume(self, new_vol: float) -> None:
        """
        default 1.0
        :param new_vol:
        :return:
        setting up volume level between 0 and 1
        """
        if 0.0 <= new_vol <= 1.0:  # 1 by default
            self.buddy.setProperty('volume', new_vol)
            self.buddy.runAndWait()
        return

    def say(self, text: str) -> None:
        """
        :param text:
        :return:
        """
        self.buddy.say(text)
        self.buddy.runAndWait()
        return

    @staticmethod
    def speak(text: str, block: bool = True) -> None:
        """
        :param text:
        :param block: if True - program waits till speaking is over
            if False - run the speak function with another process
        :return:
        static use with default params
        """
        if block:
            pyttsx3.speak(text)
        else:
            tts_path = get_file_name()
            # print(tts_path)
            # print(os.path.exists(tts_path))
            # subprocess.call([r"C:\Users\GiladEiniKbyLake\.conda\envs\wu\python.exe", "tts.py", phrase])
            # subprocess.run([r"C:\Users\GiladEiniKbyLake\.conda\envs\wu\python.exe", "tts.py", phrase])
            subprocess.Popen([sys.executable, tts_path, text])  # Call subprocess
            # args = [tts_path, '--text', text]
            # p = subprocess.Popen([sys.executable or 'python'] + args)
            # p.wait()
        return

    def get_current_property(self, key):
        val = None
        if key == 'voice_idx':
            val = self.current_voice_ind
        elif key == 'voice_name':
            voice = self.buddy.getProperty('voice')
            # noinspection PyTypeChecker
            val = os.path.basename(voice)
        elif key == 'rate':
            val = self.buddy.getProperty('rate')
        elif key == 'volume':
            val = self.buddy.getProperty('volume')
        elif key == 'all_voices':
            val = self.buddy.getProperty('voices')
        return val


try:
    # noinspection PyPackageRequirements
    from PyQt5 import QtCore, QtGui, QtWidgets  # pip install PyQt5
    from wizzi_utils.pyplot import pyplot_tools as pyplt

    # noinspection PyTypeChecker,PyUnresolvedReferences
    class MachineBuddyGui(QtWidgets.QMainWindow):
        def __init__(self, voice_idx: int = 0, rate: int = 200, volume: int = 100, loc: str = 'bl',
                     common_sentences: list = None):
            """
            :param voice_idx: index of voice from the voices installed on your device
                see machine_buddy.get_all_voices(ack=True)
            :param rate: voice rate
            :param volume: voice volume from 0 to 100
            :param loc: bl -> bottom left
            :param common_sentences: list of sentences. will be added in a combo box for easy access
            """
            super().__init__()

            self.machine_buddy = MachineBuddy(
                voice_ind=voice_idx,
                rate=rate,
                vol=volume / 100
            )

            # get the values from machine_buddy incase an illegal value was inserted
            voice_idx = self.machine_buddy.get_current_property(key='voice_idx')
            volume = int(self.machine_buddy.get_current_property(key='volume') * 100)
            rate = self.machine_buddy.get_current_property(key='rate')

            # self.app = QtWidgets.QApplication(sys.argv)
            self.setWindowTitle('MachineBuddyGui')
            w, h = 640, 215
            self.resize(w, h)
            self.move_window(self, win_loc=loc, fig_w=w, fig_h=h)
            grid = QtWidgets.QGridLayout()
            # grid.setHorizontalSpacing(0)
            # grid.setVerticalSpacing(0)
            # grid.setSpacing(0)

            # widget.setFocusPolicy(Qt.NoFocus)

            row = 0
            self.cur_settings_title_label = QtWidgets.QLabel('Current settings:')
            self.cur_settings_voice_fmt = 'voice({})={}'
            self.cur_settings_voice = QtWidgets.QLabel('')  # will be set after voice str calculated
            self.cur_settings_voice.setStyleSheet('border: 1px solid red')
            self.cur_settings_rate_vol_fmt = 'rate={}, vol={}%'
            self.cur_settings_rate_vol = QtWidgets.QLabel(self.cur_settings_rate_vol_fmt.format(rate, volume))
            self.cur_settings_rate_vol.setStyleSheet('border: 1px solid red')
            grid.addWidget(self.cur_settings_title_label, row, 0)
            grid.addWidget(self.cur_settings_voice, row, 1)
            grid.addWidget(self.cur_settings_rate_vol, row, 2)

            row = 1
            self.voices_label = QtWidgets.QLabel('Voices:')
            # self.voices_label.setStyleSheet('border: 1px solid red')
            self.voices_combo = QtWidgets.QComboBox()
            all_voices = self.machine_buddy.get_all_voices(ack=False)
            self.all_voices_name_to_idx = {}
            for i, v in enumerate(all_voices):
                self.all_voices_name_to_idx[v.name] = i
            self.voices_combo.addItems(list(self.all_voices_name_to_idx.keys()))
            self.voices_combo.setCurrentIndex(voice_idx)
            self.voices_update = QtWidgets.QPushButton('Update')
            self.voices_update.clicked.connect(self.update_voice)
            grid.addWidget(self.voices_label, row, 0)
            grid.addWidget(self.voices_combo, row, 1)
            grid.addWidget(self.voices_update, row, 2)

            # set current setting active voice
            self.cur_settings_voice.setText(
                self.cur_settings_voice_fmt.format(voice_idx, self.voices_combo.currentText()))

            row = 2
            self.volume_label = QtWidgets.QLabel('Volume {:3}%'.format(volume))
            self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            # self.volume_slider.setMaximumWidth(1000)
            # self.volume_slider.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.volume_slider.setRange(0, 100)
            self.volume_slider.setValue(volume)
            self.volume_slider.valueChanged.connect(self.update_volume_label)
            self.volume_update = QtWidgets.QPushButton('Update')
            self.volume_update.clicked.connect(self.update_volume)
            grid.addWidget(self.volume_label, row, 0)
            grid.addWidget(self.volume_slider, row, 1)
            grid.addWidget(self.volume_update, row, 2)

            row = 3
            self.rate_label = QtWidgets.QLabel('rate:')
            self.rate_line_edit = QtWidgets.QLineEdit()
            self.rate_line_edit.setText('{}'.format(rate))
            self.rate_update = QtWidgets.QPushButton('Update')
            self.rate_update.clicked.connect(self.update_rate)
            grid.addWidget(self.rate_label, row, 0)
            grid.addWidget(self.rate_line_edit, row, 1)
            grid.addWidget(self.rate_update, row, 2)

            row = 4
            self.text_to_say_label = QtWidgets.QLabel('Text to say:')
            self.text_to_say_line_edit = QtWidgets.QLineEdit()
            self.text_to_say_line_edit.setText('Shalom')
            self.text_to_say_line_edit.returnPressed.connect(self.text_to_say_button_clicked)
            self.text_to_say_button = QtWidgets.QPushButton('Say')
            # self.text_to_say_speak.setFixedWidth(40)
            self.text_to_say_button.clicked.connect(self.text_to_say_button_clicked)
            grid.addWidget(self.text_to_say_label, row, 0)
            grid.addWidget(self.text_to_say_line_edit, row, 1)
            grid.addWidget(self.text_to_say_button, row, 2)

            row = 5
            self.say_again_label = QtWidgets.QLabel('Last sentence:')
            self.say_again_line_edit = QtWidgets.QLineEdit()
            self.say_again_line_edit.setText('')
            # self.say_again_line_edit.setDisabled(True)
            # self.say_again_line_edit.returnPressed.connect(self.say_again_clicked)
            self.say_again_button = QtWidgets.QPushButton('Say again')
            # self.text_to_say_speak.setFixedWidth(40)
            self.say_again_button.clicked.connect(self.say_again_button_clicked)
            grid.addWidget(self.say_again_label, row, 0)
            grid.addWidget(self.say_again_line_edit, row, 1)
            grid.addWidget(self.say_again_button, row, 2)

            row = 6
            self.common_label = QtWidgets.QLabel('Common:')
            self.common_combo = QtWidgets.QComboBox()
            common_base = [
                'Hello',
            ]
            self.common_combo.addItems(common_base)
            if common_sentences is not None:
                self.common_combo.addItems(common_sentences)
            self.common_say_button = QtWidgets.QPushButton('Say')
            self.common_say_button.clicked.connect(self.common_say_button_clicked)

            grid.addWidget(self.common_label, row, 0)
            grid.addWidget(self.common_combo, row, 1)
            # grid.addWidget(self.common_combo, row, 1, 1, 2)  # rowspan 1 and colspan 2
            grid.addWidget(self.common_say_button, row, 2)

            main_widget = QtWidgets.QWidget()
            main_widget.setLayout(grid)
            self.setCentralWidget(main_widget)
            self.show()

            return

        def update_voice(self):
            new_voice = self.voices_combo.currentText()
            new_voice_idx = self.all_voices_name_to_idx[new_voice]
            self.machine_buddy.change_voice(new_voice_ind=new_voice_idx)
            print('voice changed to {} idx {}'.format(new_voice, new_voice_idx))
            self.cur_settings_voice.setText(  # current voice
                self.cur_settings_voice_fmt.format(
                    new_voice_idx, new_voice
                )
            )
            return

        def update_volume_label(self):
            self.volume_label.setText('Volume {:03}%'.format(self.volume_slider.value()))
            return

        def update_volume(self):
            new_vol = self.volume_slider.value() / 100
            self.machine_buddy.change_volume(new_vol=new_vol)
            print('volume changed to {}'.format(new_vol))
            self.cur_settings_rate_vol.setText(
                self.cur_settings_rate_vol_fmt.format(
                    self.machine_buddy.get_current_property(key='rate'), self.volume_slider.value())
            )
            return

        def update_rate(self):
            new_rate = self.rate_line_edit.text()
            if new_rate.isnumeric():
                new_rate = int(new_rate)
                self.machine_buddy.change_rate(new_rate=new_rate)
                print('rate changed to {}'.format(new_rate))
                self.cur_settings_rate_vol.setText(
                    self.cur_settings_rate_vol_fmt.format(
                        new_rate, int(self.machine_buddy.get_current_property(key='volume') * 100))
                )
            else:
                print('value {} must be int'.format(new_rate))
            return

        def text_to_say_button_clicked(self):
            text_to_say = self.text_to_say_line_edit.text()
            if text_to_say:
                self.text_to_say_line_edit.clear()
                print('saying {}'.format(text_to_say))
                self.machine_buddy.say(text=text_to_say)
                self.say_again_line_edit.setText(text_to_say)
            return

        def say_again_button_clicked(self):
            text_to_say_again = self.say_again_line_edit.text()
            if text_to_say_again:
                print('saying again {}'.format(text_to_say_again))
                self.machine_buddy.say(text=text_to_say_again)
            return

        def common_say_button_clicked(self):
            common_say = self.common_combo.currentText()
            if common_say:
                print('common say {}'.format(common_say))
                self.machine_buddy.say(text=common_say)
            return

        def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() in [QtCore.Qt.Key_Escape]:
                print('keyPressEvent(): Qt plot: Esc was clicked. Terminating...')
                MachineBuddyGui.get_qt_app().quit()

        @staticmethod
        def get_qt_app() -> QtWidgets.QApplication:
            app = QtWidgets.QApplication.instance()  # main app must init before creating GLViewWidget
            if QtWidgets.QApplication.instance() is None:
                app = QtWidgets.QApplication(sys.argv)
            return app

        def run(self):
            return sys.exit(self.app.exec_())

        @staticmethod
        def move_window(widget, win_loc, fig_w, fig_h):
            # move window
            window_w, window_h = pyplt.screen_dims()  # screen dims in pixels
            taskbar_size = 70
            x, y = None, None
            if win_loc == 'tl':  # top left
                x, y = 0, 0
            elif win_loc == 'tc':  # top center
                x, y = (window_w - fig_w) / 2, 0
            elif win_loc == 'tr':  # top right
                x, y = window_w - fig_w, 0
            elif win_loc == 'cl':  # center left
                x, y = 0, (window_h - fig_h) / 2
            elif win_loc == 'cc':  # center center
                x, y = (window_w - fig_w) / 2, (window_h - fig_h) / 2
            elif win_loc == 'cr':  # center right
                x, y = window_w - fig_w, (window_h - fig_h) / 2 - taskbar_size
            elif win_loc == 'bl':  # bottom left
                x, y = 0, window_h - fig_h - taskbar_size
            elif win_loc == 'bc':  # bottom center
                x, y = (window_w - fig_w) / 2, window_h - fig_h - taskbar_size
            elif win_loc == 'br':  # bottom right
                x, y = window_w - fig_w, window_h - fig_h - taskbar_size
            widget.move(x, y)
            return


    def run_machine_buddy_gui(
            voice_idx: int = 0,
            rate: int = 200,
            volume: int = 100,
            loc: str = 'bl',
            common_sentences: list = None
    ) -> None:
        """see class MachineBuddyGui documentation"""
        app = MachineBuddyGui.get_qt_app()
        # noinspection PyUnusedLocal
        mbg = MachineBuddyGui(voice_idx=voice_idx, rate=rate, volume=volume, loc=loc, common_sentences=common_sentences)
        return sys.exit(app.exec_())

except (ModuleNotFoundError, ImportError):
    pass
