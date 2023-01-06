import wizzi_utils as wu


def main():
    # TODO 1 change images on md file to this repo
    # TODO 2 md file add repo link
    # TODO 3 why req file not appears at user side
    # TODO 4 check warning in reqs: update to safe reqs
    #           update req txt
    wu.got.test.get_link_test()
    # todo bug fix repo_root_path = get_file_name(depth=2)
    
    # wu.test_all_modules()
    # wu.test.generate_requirements_file_test(real_req=True)
    # wu.wizzi_utils_requirements()

    # wu.test.test_all()  # misc package
    # # wu.got.test.test_all()  # google package
    # wu.jt.test.test_all()  # json package
    # wu.cvt.test.test_all()  # open_cv package
    # wu.pyplt.test.test_all()  # pyplot package
    # wu.st.test.test_all()  # socket package
    # wu.tt.test.test_all()  # torch package
    # wu.tflt.test.test_all()  # tflite package
    # wu.tts.test.test_all()  # tts package
    # wu.tts.test.run_machine_buddy_gui_test()
    # wu.models.test.test_all()  # models package
    return


if __name__ == '__main__':
    wu.main_wrapper(
        main_function=main,
        seed=42,
        ipv4=True,
        cuda_off=False,
        torch_v=True,
        tf_v=True,
        cv2_v=True,
        with_pip_list=False,
        with_profiler=False
    )
