import wizzi_utils as wu


def main():
    # wu.test_all_modules()
    # wu.generate_requirements_file(fp_out='./my_req.txt', ack=False)
    # wu.download_wizzi_utils_env_snapshot()
    # wu.download_wizzi_utils_requirements_txt()

    # wu.test.test_all()  # misc package
    # wu.got.test.test_all()  # google package  # TODO WIP
    # wu.jt.test.test_all()  # json package
    # wu.cvt.test.test_all()  # open_cv package
    # wu.pyplt.test.test_all()  # pyplot package
    # wu.st.test.test_all()  # socket package
    # wu.tt.test.test_all()  # torch package
    # wu.tflt.test.test_all()  # tflite package
    # wu.tts.test.test_all()  # tts package
    # TODO ImportError: cannot import name 'QtCore' from 'PyQt5' (unknown location)
    # wu.tts.test.run_machine_buddy_gui_test()
    # wu.models.test.test_all()  # models package
    return


if __name__ == '__main__':
    wu.main_wrapper(
        main_function=main,
        seed=42,
        ipv4=True,
        cuda_off=False,
        nvid_gpu=True,
        torch_v=True,
        tf_v=True,
        cv2_v=True,
        with_pip_list=False,
        with_profiler=False
    )
