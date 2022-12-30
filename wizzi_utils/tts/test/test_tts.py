from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.tts.tts import MachineBuddy


def str_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    machine_buddy = MachineBuddy()
    # machine_buddy.say('hello')
    print(machine_buddy)
    machine_buddy.change_voice(1)
    machine_buddy.change_rate(250)
    machine_buddy.change_volume(0.5)
    print(machine_buddy)
    # machine_buddy.say('hello')
    return


def instance_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    print('\taudio test')
    machine_buddy = MachineBuddy(
        voice_ind=0,
        rate=150,
        vol=1.0
    )
    machine_buddy.say('instance test')
    return


def static_block_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    print('\taudio test')
    MachineBuddy.speak('static block test', block=True)  # Static use:
    return


def voices_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    machine_buddy = MachineBuddy(rate=150)
    all_voices = machine_buddy.get_all_voices(ack=True)

    print('\taudio test')
    for i, v in enumerate(all_voices):
        machine_buddy.change_voice(new_voice_ind=i)
        machine_buddy.say(text=v.name)
        if 'Hebrew' in str(v.name):
            t = 'שלום, מה קורה חברים?'
            machine_buddy.say(text=t)
    return


def vol_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    machine_buddy = MachineBuddy()
    vol_options = [0.3, 0.6, 0.9]
    print('\taudio test')
    for vol in vol_options:
        machine_buddy.change_volume(new_vol=vol)
        machine_buddy.say(text='volume at {}%'.format(int(vol * 100)))
    return


def rate_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    machine_buddy = MachineBuddy()
    rate_options = [100, 300, 500]
    print('\taudio test')
    for rate in rate_options:
        machine_buddy.change_rate(new_rate=rate)
        machine_buddy.say(text='rate is {}'.format(rate))
    return


def static_without_block_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    print('\taudio test')
    MachineBuddy.speak('static without block test', block=False)  # Static use:
    return


def run_machine_buddy_gui_test():
    try:
        # noinspection PyUnresolvedReferences
        from wizzi_utils.tts.tts import run_machine_buddy_gui
        mt.get_function_name_and_line(ack=True, tabs=0)
        cs = [
            'I say this a lot of times',
            'what\'s cooking good looking'
        ]
        run_machine_buddy_gui(voice_idx=0, rate=150, volume=90, loc='tl', common_sentences=cs)
    except ImportError as e:
        mt.exception_error(e, real_exception=True)
        mt.exception_error('try pip install PyQt5')
        mt.exception_error('or pip install matplotlib')
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name(depth=1)))
    str_test()
    instance_test()
    static_block_test()
    voices_test()
    vol_test()
    rate_test()
    static_without_block_test()
    print('{}'.format('-' * 20))
    return
