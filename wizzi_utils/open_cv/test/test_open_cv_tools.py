from wizzi_utils.misc import misc_tools as wu
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.socket import socket_tools as st
from wizzi_utils.pyplot import pyplot_tools as pyplt
import numpy as np
import os
from collections import OrderedDict
# noinspection PyPackageRequirements
import cv2

LOOP_TESTS = 50
BLOCK_MS_NORMAL = 2000  # 0 to block
ITERS_CAM_TEST = 10  # 0 to block


def load_img_from_web(name: str, ack: bool = True) -> np.array:
    f = mtt.IMAGES_INPUTS
    url = mtt.IMAGES_D[name]
    suffix = 'jpg'  # default
    # if '.webm' in url:
    #     suffix = 'webm'
    dst = '{}/{}.{}'.format(f, name, suffix)

    if not os.path.exists(dst):
        if not os.path.exists(f):
            wu.create_dir(f)
        success = st.download_file(url, dst)
        if not success:
            wu.exception_error('download failed - creating random img', real_exception=False)
            img = wu.np_random_integers(size=(240, 320, 3), low=0, high=255)
            img = img.astype('uint8')
            cvt.save_img(dst, img)

    img = cvt.load_img(path=dst, ack=ack)
    return img


def get_vid_from_web(name: str) -> str:
    f = mtt.VIDEOS_INPUTS
    url = mtt.VIDEOS_D[name]
    suffix = 'mp4'  # default
    if '.webm' in url:
        suffix = 'webm'
    dst = '{}/{}.{}'.format(f, name, suffix)

    if not os.path.exists(dst):
        if not os.path.exists(f):
            wu.create_dir(f)
        success = st.download_file(url, dst)
        if not success:
            wu.exception_error('download failed - creating random img', real_exception=False)
            dst = None

    return dst


def get_cv_version_test():
    wu.get_function_name(ack=True, tabs=0)
    cvt.get_cv_version(ack=True, tabs=1)
    return


def imread_imwrite_test():
    wu.get_function_name(ack=True, tabs=0)
    name = mtt.SO_LOGO
    img = load_img_from_web(name)

    f = mtt.IMAGES_INPUTS
    url = mtt.IMAGES_D[name]
    dst_path = '{}/{}'.format(f, os.path.basename(url).replace('.png', '_copy.png'))

    cvt.save_img(dst_path, img, ack=True)
    img_loaded = cvt.load_img(dst_path, ack=True)
    print(wu.to_str(img_loaded, '\timg_copy'))
    wu.delete_file(dst_path, ack=True)

    dst_path = '{}'.format(os.path.basename(url).replace('.png', '_copy.png'))  # no dir - local folder
    cvt.save_img(dst_path, img, ack=True)
    wu.delete_file(dst_path, ack=True)

    dst_path = 'NoSuchDir/{}'.format(os.path.basename(url).replace('.png', '_copy.png'))  # bad folder
    cvt.save_img(dst_path, img, ack=True)

    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def list_to_cv_image_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    img_list = img.tolist()
    print(wu.to_str(img_list, '\timg_list'))
    img = cvt.list_to_cv_image(img_list)
    print(wu.to_str(img, '\timg'))
    # wu.delete_file(file=mtt.TEMP_IMAGE_PATH, ack=True)
    return


def display_open_cv_image_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    print('\tVisual test: stack overflow logo')
    loc = (70, 200)  # move to X,Y
    resize = 1.7  # enlarge to 170%
    cvt.display_open_cv_image(
        img=img,
        ms=1,  # not blocking
        title='stack overflow logo moved to {} and re-sized to {}'.format(loc, resize),
        loc=loc,  # start from x =70 y = 0
        resize=resize
    )
    loc = pyplt.Location.TOP_RIGHT.value  # move to top right corner
    resize = 1.7  # enlarge to 170%
    cvt.display_open_cv_image(
        img=img,
        ms=BLOCK_MS_NORMAL,  # blocking
        title='stack overflow logo moved to {} and re-sized to {}'.format(loc, resize),
        loc=loc,  # start from x =70 y = 0
        resize=resize
    )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def display_open_cv_image_loop_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    loc = (70, 200)  # move to X,Y
    resize = 1.7  # enlarge to 170%
    title = 'stack overflow logo moved to {} and re-sized to {} - {} iterations'.format(loc, resize, LOOP_TESTS)
    print('\tVisual test: {}'.format(title))
    for i in range(LOOP_TESTS):
        cvt.display_open_cv_image(
            img=img,
            ms=1,  # not blocking
            title=title,
            loc=loc if i == 0 else None,  # move just first iter
            resize=resize
        )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def resize_opencv_image_test():
    wu.get_function_name(ack=True, tabs=0)
    img1 = load_img_from_web(mtt.SO_LOGO)
    img2 = img1.copy()
    img3 = img1.copy()

    cvt.display_open_cv_image(
        img=img1,
        ms=1,
        title='int test',
        loc='tl',
        resize=2,
        header='resize to 200%',
        save_path=None,
    )

    cvt.display_open_cv_image(
        img=img1,
        ms=1,
        title='float >1 test',
        loc='cl',
        resize=1.6,
        header='resize to 160%',
        save_path=None,
    )

    cvt.display_open_cv_image(
        img=img1,
        ms=1,
        title='float <1 test',
        loc='tc',
        resize=0.7,
        header='resize to 70%',
        save_path=None,
    )

    cvt.display_open_cv_image(
        img=img2,
        ms=BLOCK_MS_NORMAL,
        title='tuple test',
        loc='bl',
        resize=(400, 200),
        header='resize to (400, 200)',
        save_path=None,
    )
    cv2.destroyAllWindows()

    # no full-screen
    cvt.display_open_cv_image(
        img=img3,
        ms=BLOCK_MS_NORMAL,
        title='img4',
        loc=pyplt.Location.BOTTOM_LEFT.value,
        resize='fs',
        header='resize to full-screen',
        save_path=None,
    )
    cv2.destroyAllWindows()
    return


def move_cv_img_x_y_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    options = [(0, 0), (100, 0), (0, 100), (150, 150), (400, 400), (250, 350)]
    print('\tVisual test: move to all options {}'.format(options))
    print('\t\tClick Esc to close all')
    for x_y in options:
        title = 'move to ({})'.format(x_y)
        cv2.imshow(title, img)
        cvt.move_cv_img_x_y(title, x_y)
    cv2.waitKey(BLOCK_MS_NORMAL)
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def move_cv_img_by_str_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    options = pyplt.Location.get_location_list_by_rows()
    print('\tVisual test: move to all options {}'.format(options))
    print('\t\tClick Esc to close all')
    for where_to in options:
        title = 'move to {}'.format(where_to)
        cv2.imshow(title, img)
        cvt.move_cv_img_by_str(img, title, where=where_to)
    cv2.waitKey(BLOCK_MS_NORMAL)
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def unpack_list_imgs_to_big_image_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    gray = cvt.bgr_to_gray(img)
    big_img = cvt.unpack_list_imgs_to_big_image(
        imgs=[img, gray, img],
        grid=(2, 2)
    )
    title = 'stack overflow logo 2x2(1 empty)'
    print('\tVisual test: {}'.format(title))
    cvt.display_open_cv_image(
        img=big_img,
        ms=BLOCK_MS_NORMAL,  # blocking
        title=title,
        loc=(0, 0),
        resize=None
    )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def display_open_cv_images_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)

    title = '3x1 grid(1 empty)'
    print('\tVisual test: {}'.format(title))
    loc1 = (0, 0)
    cvt.display_open_cv_images(
        imgs=[img, img],
        ms=1,
        title='{} loc={}'.format(title, loc1),
        loc=loc1,
        resize=1.5,
        grid=(3, 1),  # one empty frame
        header='{} loc={}'.format(title, loc1),
        separator_c='aqua',
        empty_img_c='orange',
        empty_img_txt='No image'
    )
    # loc2 = pyplt.Location.BOTTOM_CENTER.value
    loc2 = 'bc'
    cvt.display_open_cv_images(
        imgs=[img, img],
        ms=BLOCK_MS_NORMAL,  # blocking
        title='{} loc={}'.format(title, loc2),
        loc=loc2,
        resize=None,
        grid=(2, 1),
        header='{} {}'.format(title, pyplt.Location.where_to(loc2)),
    )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def display_open_cv_images_loop_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    loc = (70, 200)  # move to X,Y
    title = 'stack overflow logo moved to {} - {} iterations'.format(loc, LOOP_TESTS)
    print('\tVisual test: {}'.format(title))
    for i in range(LOOP_TESTS):
        cvt.display_open_cv_images(
            imgs=[img, img],
            ms=1,  # blocking
            title=title,
            loc=loc if i == 0 else None,  # move just first iter
            resize=None,
            grid=(2, 1),
            header=None
        )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def gray_to_BGR_and_back_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.SO_LOGO)
    print(wu.to_str(img, '\timgRGB'))
    gray = cvt.bgr_to_gray(img)
    print(wu.to_str(img, '\timg_gray'))
    img = cvt.gray_to_bgr(gray)
    print(wu.to_str(img, '\timgRGB'))
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def BGR_img_to_RGB_and_back_test():
    wu.get_function_name(ack=True, tabs=0)
    imgBGR1 = load_img_from_web(mtt.SO_LOGO)
    print(wu.to_str(imgBGR1, '\timgBGR'))
    imgRGB = cvt.bgr_to_rgb(imgBGR1)
    print(wu.to_str(imgRGB, '\timgRGB'))
    imgBGR2 = cvt.rgb_to_bgr(imgRGB)
    print(wu.to_str(imgBGR2, '\timgBGR2'))

    cvt.display_open_cv_images(
        imgs=[imgBGR1, imgRGB, imgBGR2],
        ms=BLOCK_MS_NORMAL,  # blocking
        title='imgBGR1, imgRGB, imgBGR2',
        loc=pyplt.Location.CENTER_CENTER,
        resize=None,
        grid=(3, 1),
        header='compare'
    )
    cv2.destroyAllWindows()
    # wu.delete_file(file=mtt.SO_LOGO_PATH, ack=True)
    return


def CameraWu_all_res_test():
    CameraWu_res_test('cv2')
    CameraWu_res_test('imutils')
    return


def CameraWu_res_test(type_cam: str):
    wu.get_function_name(ack=True, tabs=0)
    port = 0
    cam = cvt.CameraWu.open_camera(port=port, type_cam=type_cam)
    if cam is not None:
        title = 'CameraWu_test({}) on port {}'.format(cam.type_cam, cam.port)
        for w, h in [(1920, 1080), (1280, 720), (640, 480), (640, 360)]:
            success_res = cam.set_resolution(w=w, h=h, ack=False)

            if success_res:
                if type_cam == 'imutils':
                    wu.sleep(2)  # allow time to change before reading frame
                success, cv_img = cam.read_img()
                for i in range(5):  # sometimes first image is black
                    success, cv_img = cam.read_img()
                if success:
                    k = cvt.display_open_cv_image(
                        img=cv_img,
                        ms=BLOCK_MS_NORMAL,
                        title=title,
                        loc=pyplt.Location.TOP_CENTER.value,
                        resize=None,
                        header='res {}x{}'.format(w, h)
                    )
                    if k == ord('q'):
                        wu.exception_error('q was clicked. breaking loop')
                        # break
                    cv2.destroyAllWindows()
    return


def CameraWu_test(type_cam: str):
    WITH_SLEEP = False
    ports = [0, 1, 13]
    cams = []
    for port in ports:
        cam = cvt.CameraWu.open_camera(port=port, type_cam=type_cam)
        if cam is not None:
            cams.append(cam)

    for cam in cams:
        title = 'CameraWu_test({}) on port {}'.format(cam.type_cam, cam.port)
        fps = wu.FPS(summary_title=title)
        for i in range(ITERS_CAM_TEST):
            fps.start()
            success, cv_img = cam.read_img()

            if WITH_SLEEP:
                wu.sleep(1)

            if success:
                k = cvt.display_open_cv_image(
                    img=cv_img,
                    ms=1,
                    title=title,
                    loc=pyplt.Location.TOP_CENTER.value,
                    resize=None,
                    header='{}/{}'.format(i + 1, ITERS_CAM_TEST)
                )
                if k == ord('q'):
                    wu.exception_error('q was clicked. breaking loop')
                    break
            fps.update()
        fps.finalize()
    cv2.destroyAllWindows()
    return


def CameraWu_cv2_test():
    wu.get_function_name(ack=True, tabs=0)
    CameraWu_test(type_cam='cv2')
    return


def CameraWu_acapture_test():
    wu.get_function_name(ack=True, tabs=0)
    CameraWu_test(type_cam='acapture')
    return


def CameraWu_imutils_test():
    wu.get_function_name(ack=True, tabs=0)
    CameraWu_test(type_cam='imutils')
    return


def add_text_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.HORSES)
    cvt.add_text(img, header='test text', pos=(100, 100), text_color='r', with_rect=True, bg_color='y', bg_font_scale=2)
    cvt.add_text(img, header='test text', pos=(100, 200), text_color='black', with_rect=True, bg_color='b',
                 bg_font_scale=1)
    cvt.display_open_cv_image(img, ms=BLOCK_MS_NORMAL, loc=pyplt.Location.CENTER_CENTER.value)
    cv2.destroyAllWindows()
    return


def add_header_test():
    wu.get_function_name(ack=True, tabs=0)
    img = load_img_from_web(mtt.HORSES)

    cvt.add_header(img, header='TOP_LEFT', loc=pyplt.Location.TOP_LEFT.value,
                   text_color='lime', with_rect=True, bg_color='azure', bg_font_scale=1)
    cvt.add_header(img, header='BOTTOM_LEFT', loc=pyplt.Location.BOTTOM_LEFT.value,
                   text_color='fuchsia', with_rect=True, bg_color='black', bg_font_scale=2)
    cvt.add_header(img, header='TOP_RIGHT', loc='tr', x_offset=180,  # tr = top right
                   text_color='darkorange', with_rect=True, bg_color='azure', bg_font_scale=1)
    cvt.add_header(img, header='BOTTOM_RIGHT', loc=pyplt.Location.BOTTOM_RIGHT.value, x_offset=120,
                   text_color='aqua', with_rect=True, bg_color='black', bg_font_scale=2)
    cvt.display_open_cv_image(img, title='all headers', ms=1, loc=pyplt.Location.TOP_LEFT.value)

    img = load_img_from_web(mtt.DOG)
    cvt.display_open_cv_image(
        img,
        title='built in header',
        ms=BLOCK_MS_NORMAL,
        loc=pyplt.Location.TOP_RIGHT.value,
        header='direct header into display_open_cv_image'
    )
    cv2.destroyAllWindows()
    return


def get_video_details_test():
    wu.get_function_name(ack=True, tabs=0)
    vid_name = mtt.DOG1
    url = mtt.VIDEOS_D[vid_name]
    video_path = get_vid_from_web(name=vid_name)
    if not os.path.exists(video_path):
        wu.exception_error(wu.NOT_FOUND.format(video_path), real_exception=False)
        return
    print('Video {} details({}):'.format(vid_name, url))
    cvt.get_video_details(video_path=video_path, ack=True)
    return


def get_video_details_imutils_test():
    try:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from imutils import video  # pip install imutils
        wu.get_function_name(ack=True, tabs=0)
        vid_name = mtt.DOG1
        url = mtt.VIDEOS_D[vid_name]
        video_path = get_vid_from_web(name=vid_name)
        if not os.path.exists(video_path):
            wu.exception_error(wu.NOT_FOUND.format(video_path), real_exception=False)
            return
        print('Video {} details({}):'.format(vid_name, url))
        cap = video.VideoStream(src=video_path).start()
        out_dims = cvt.get_dims_from_cap_imutils(cap)
        video_total_frames = cvt.get_frames_from_cap_imutils(cap)
        fps = cvt.get_fps_from_cap_imutils(cap)
        duration = cvt.get_video_duration_from_cap_imutils(cap)
        size = wu.file_or_folder_size(path=video_path)
        print('Video {} details:'.format(os.path.abspath(video_path)))
        print('\t{} frames'.format(video_total_frames))
        print('\tframe size {}'.format(out_dims))
        print('\t{} fps'.format(fps))
        print('\tduration {} seconds'.format(duration))
        print('\tfile size {}'.format(size))
        cap.stop()
    except ModuleNotFoundError:
        pass
    return


def video_creator_mp4_test():
    wu.get_function_name(ack=True, tabs=0)
    # now open video file
    vid_name = mtt.DOG1
    video_path = get_vid_from_web(name=vid_name)

    if not os.path.exists(video_path):
        wu.exception_error(wu.NOT_FOUND.format(video_path), real_exception=False)
        return

    out_dims, video_total_frames, fps, duration, size = cvt.get_video_details(video_path=video_path, ack=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        wu.exception_error('cap is closed.', real_exception=False)
        return

    out_dir = '{}/create_mp4_test'.format(mtt.VIDEOS_OUTPUTS)
    wu.create_dir(out_dir)
    # out_fp = '{}/{}_output.mp4'.format(out_dir, vid_name)
    out_fp = '{}/{}_output'.format(out_dir, vid_name)  # not specifying extension

    mp4_creator = cvt.VideoCreator(
        out_full_path=out_fp,  # since coded is mp4v, file will get .mp4 extension
        out_fps=20.0,
        out_dims=out_dims,
        codec='mp4v',
        tabs=1
    )
    print(mp4_creator)

    for i in range(video_total_frames):
        success, frame = cap.read()
        if i % int(video_total_frames / 10) != 0:  # s
            # do only 10 frames
            continue
        # print('\tframe {}/{}:'.format(i + 1, video_total_frames))
        # print('\t\t{}'.format(wu.to_str(frame)))
        if success:
            cvt.add_header(
                frame,
                header='create_mp4_test frame {}/{}'.format(i + 1, video_total_frames),
                loc=pyplt.Location.BOTTOM_LEFT.value,
                text_color=pyplt.get_random_color(),
                bg_color=pyplt.get_random_color(),
            )
            cvt.display_open_cv_image(frame, ms=1, title=vid_name, loc=None,
                                      header='{}/{}'.format(i + 1, video_total_frames))
            mp4_creator.add_frame(frame, ack=False, tabs=2)

    cap.release()
    mp4_creator.finalize()
    cv2.destroyAllWindows()
    return


def video_creator_size_test():
    wu.get_function_name(ack=True, tabs=0)

    if wu.is_windows():  # fix for mkv files
        # dll for H264 and X264 codecs for python 3.8
        # set this to the dir of the dll
        mkv_dll_path = r'{}/resources'.format(wu.get_repo_root(repo_name='wizzi_utils_pypi'))
        if not os.path.exists(mkv_dll_path):
            print('not found {}'.format(os.path.abspath(mkv_dll_path)))
            exit(-9)
        path_env = wu.get_env_variable(key='PATH')
        path_env = '{};{}'.format(path_env, os.path.abspath(mkv_dll_path))  # add mkv_dll_path
        wu.set_env_variable(key='PATH', val=path_env)
        # wu.get_env_variable(key='PATH', ack=True)

    wu.add_resource_folder_to_path(resources_dir='../resources')
    # now open video file
    vid_name = mtt.DOG1
    video_path = get_vid_from_web(name=vid_name)

    if not os.path.exists(video_path):
        wu.exception_error(wu.NOT_FOUND.format(video_path), real_exception=False)
        return

    out_dims, video_total_frames, fps, duration, size = cvt.get_video_details(video_path=video_path, ack=True)
    out_dir = '{}/video_creator_size_test'.format(mtt.VIDEOS_OUTPUTS)
    wu.create_dir(out_dir)
    codecs = list(cvt.VideoCreator.CODECS_TO_EXT.keys())
    sizes = []
    for codec in codecs:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            wu.exception_error('cap is closed.', real_exception=False)
            return

        out_fp = '{}/{}_output_{}'.format(out_dir, vid_name, codec)  # not specifying extension
        mp4_creator = cvt.VideoCreator(
            out_full_path=out_fp,  # since coded is mp4v, file will get .mp4 extension
            out_fps=20.0,
            out_dims=out_dims,
            codec=codec,
            tabs=1
        )
        # print(mp4_creator)

        for i in range(video_total_frames):
            success, frame = cap.read()
            if i % int(video_total_frames / 10) != 0:  # s
                # do only 10 frames
                continue
            if success:
                cvt.add_header(
                    frame,
                    header='create_mp4_test frame {}/{}'.format(i + 1, video_total_frames),
                    loc=pyplt.Location.BOTTOM_LEFT.value,
                    text_color=pyplt.get_random_color(),
                    bg_color=pyplt.get_random_color(),
                )
                cvt.display_open_cv_image(frame, ms=1, title=vid_name, loc=None,
                                          header='{}/{}'.format(i + 1, video_total_frames))
                mp4_creator.add_frame(frame, ack=False, tabs=2)

        cap.release()
        mp4_creator.finalize()
        file_size = wu.file_or_folder_size(path=mp4_creator.out_full_path)
        sizes.append(file_size)
        cv2.destroyAllWindows()

    print('codes to file size summary:')
    for codec, file_size in zip(codecs, sizes):
        print('\t{}: {}'.format(codec, file_size))
    wu.delete_dir_with_files(out_dir)
    return


def get_aspect_ratio_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img_fake = np.zeros(shape=(480, 640, 3))
    img_h, img_w = cv_img_fake.shape[0], cv_img_fake.shape[1]
    new_h = 192
    resize_dims = cvt.get_aspect_ratio_w(img_w=img_w, img_h=img_h, new_h=new_h)
    print('\timage size={} and new_h={}: new dims should be {}'.format(cv_img_fake.shape, new_h, resize_dims))

    new_w = 192
    resize_dims = cvt.get_aspect_ratio_h(img_w=img_w, img_h=img_h, new_w=new_w)
    print('\timage size={} and new_w={}: new dims should be {}'.format(cv_img_fake.shape, new_w, resize_dims))
    return


def cuda_on_gpu_test():
    """
    opencv cuda installation on WINDOWS 10:
    youtube https://www.youtube.com/watch?v=YsmhKar8oOc&ab_channel=TheCodingBug
    ** all links for download in the link
    ** in brackets: my specific installation
    * install anaconda
    * vs studio 19 with "desktop development C++" and "python development"
    * Cuda toolkit (cuda toolkit 10.2)
    * cuDNN - version that match the cuda toolkit above (cuDnn v7.6.5 for Cuda 10.2)
        * archive: https://developer.nvidia.com/rdp/cudnn-archive
        * copy all to cuda (bin to bin, lib to lib, include to include)
        * minimum 7.5
    * CMake
    * openCV source (4.5.2)
    * openCV contrib - same version as openCV above (4.5.2)
        * place openCV and openCV contrib in the same folder and extract both

    1 build a new python env anywhere(conda(base or custom), normal py, venv...)
    2 set 4 system environment variables:
        let PY_PATH be you python dir (conda env, normal py, venv...)
        go to PATH and add the next 4:
            * PY_PATH # for the python.exe
            * PY_PATH/Scripts # for pip...
            * PY_PATH/libs  # for python36.lib file
            * PY_PATH/include # h files
        e.g. added to path system variable:
            C:/Users/GiladEiniKbyLake/.conda/envs/py385cvCuda
            C:/Users/GiladEiniKbyLake/.conda/envs/py385cvCuda/Scripts
            C:/Users/GiladEiniKbyLake/.conda/envs/py385cvCuda/include
            C:/Users/GiladEiniKbyLake/.conda/envs/py385cvCuda/libs
    2b pip install numpy (if you want tf, 1.19.5)
    3 open CMake gui
    * first delete cache (file > delete cache)
    * source code: location of openCV source
    * create 2 new dirs in the level of openCV source called build and install
    * where to build: location of build folder created above
    * configure, set generator x64 and finish
    * click grouped checkbox and look in PYTHON3
        should have EXECUTABLE, INCLUDE_DIR, LIBRARY, NUMPY_INCLUDE, PACKAGES paths filled according to PY_PATH
        look at the output and search for OpenCV modules. if python3 in Unavailable - don't continue. it should be
        in the "To be built"
    * extra round of flags:
        WITH_CUDA
        BUILD_opencv_dnn
        OPENCV_DNN_CUDA
        ENABLE_FAST_MATH
        BUILD_opencv_world
        BUILD_opencv_python3 # should already be on
        OPENCV_EXTRA_MODULES_PATH -> set to openCV contrib on modules folder
    * hit configure - in the output you should see your CUDA and cuDNN details
        CUDA detected: x.y (10.2)
        Found CUDNN path ...
    * extra round of flags:
        CUDA_FAST_MATH
        CUDA_ARCH_BIN -> go to https://en.wikipedia.org/wiki/CUDA and look for your GPU. find it's Compute
            capability (version). (my GPU is gtx 1050. version 6.1)
            remove all versions other than you GPU version. (i left only 6.1)
        CMAKE_INSTALL_PREFIX -> place install path from above (you create this dir with build above)
        CMAKE_CONFIGURATION_TYPES -> remove Debug and keep only Release
    * hit configure
    * hit generate
    close CMAKE gui
    4 go to build folder and look for OpenCV.sln
    * goto solution explorer->CMakeTargets-> right click ALL_BUILD and hit build # should take 10-30 mins
    * if done with no errors, right click INSTALL and build.
    * if no errors, openCV cuda is ready
    5 validation: open terminal and write python (should work due to step 2)
    import cv2
    print(cv2.__version__)
    print(cv2.cuda.getCudaEnabledDeviceCount())  # should be >=1
    run this:
        pip install wizzi_utils
        import wizzi_utils as wu
        wu.cvt.test.cuda_on_gpu_test()
    6 cleanup
    * delete all build folder except build/lib # maybe for future use ?
    * delete both zips and extracted open cv and open cv contrib
    7 after checking on pycharm, open cv with GPU support should work but no autocomplete.
    pip install mypy
    stubgen -m cv2 -o ENV_PATH/Lib/site-packages/cv2
    # if the stubgen fails, run as administrator
    # rename cv2.pyi created to __init__.pyi
    # all done
    e.g:
    stubgen -m cv2 -o C:\\Users\\GiladEiniKbyLake\\.conda\\envs\\cv_cuda\\Lib\\site-packages\\cv2
    # a file was created at C:\\Users\\GiladEiniKbyLake\\.conda\\envs\\temp\\Lib\\site-packages\\cv2\\cv2.pyi
    # rename cv2.pyi to __init__.pyi and have the file:
        C:\\Users\\GiladEiniKbyLake\\.conda\\envs\\temp\\Lib\\site-packages\\cv2\\__init__.pyi
    :return:
    """
    wu.get_function_name(ack=True, tabs=0)
    print(cvt.get_cv_version())
    gpus = cvt.get_gpu_devices_count()
    print('\t{} GPU devices detected'.format(gpus))
    if gpus > 0:
        npTmp = np.random.random((1024, 1024)).astype(np.float32)

        npMat1 = np.stack([npTmp, npTmp], axis=2)
        npMat2 = npMat1

        cuMat1 = cv2.cuda_GpuMat()
        cuMat2 = cv2.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)
        # start_time = time.time()
        start_time = wu.get_timer()
        # noinspection PyUnresolvedReferences
        cv2.cuda.gemm(cuMat1, cuMat2, 1, None, 0, None, 1)
        print("\t\tCUDA --- %s seconds ---" % (wu.get_timer() - start_time))
        # start_time = time.time()
        start_time = wu.get_timer()

        cv2.gemm(npMat1, npMat2, 1, None, 0, None, 1)
        print("\t\tCPU  --- %s seconds ---" % (wu.get_timer() - start_time))
    return


def sketch_image_test():
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')
    cv_img = load_img_from_web(mtt.DOG, ack=False)
    skg, skc = cvt.sketch_image(cv_bgr=cv_img)

    cvt.add_header(skg, header='sketch_gray')
    cvt.add_header(skc, header='sketch_color')

    cvt.display_open_cv_images(
        imgs=[cv_img, skg, skc],
        ms=BLOCK_MS_NORMAL,
        title='sketch_image_test',
        loc='tl',
        resize=0.7,
        grid=(1, 3),
        header=None
    )

    cv2.destroyAllWindows()
    return


def invert_image_test():
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')

    cv_img = load_img_from_web(mtt.DOG, ack=False)
    invert_bgr = cvt.invert_image(cv_img=cv_img)
    cvt.add_header(invert_bgr, header='invert_bgr')

    cv_gray = cvt.bgr_to_gray(cv_img)
    invert_gray = cvt.invert_image(cv_img=cv_gray)
    cvt.add_header(invert_gray, header='invert_gray')

    cvt.display_open_cv_images(
        imgs=[cv_img, invert_bgr, invert_gray],
        ms=BLOCK_MS_NORMAL,
        title='invert_image_test',
        loc='tl',
        resize=0.7,
        grid=(1, 3),
        header=None
    )

    cv2.destroyAllWindows()
    return


def add_color_map_test():
    # https://learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')
    cv_img = load_img_from_web(mtt.DOG, ack=False)
    cv_gray = cvt.bgr_to_gray(cv_img)
    color_map = cv2.COLORMAP_PLASMA

    colored_img_on_bgr = cvt.add_color_map(cv_img=cv_img, color_map=color_map)
    cvt.add_header(colored_img_on_bgr, header='bgr base: color idx {}'.format(color_map))

    colored_img_on_gray = cvt.add_color_map(cv_img=cv_gray, color_map=color_map)
    cvt.add_header(colored_img_on_gray, header='gray base: color idx {}'.format(color_map))

    cvt.display_open_cv_images(
        imgs=[cv_img, colored_img_on_bgr, colored_img_on_gray],
        ms=BLOCK_MS_NORMAL,
        title='add_colors_to_gray_image_test',
        loc='tl',
        resize=0.7,
        grid=(1, 3),
        header=None
    )
    cv2.destroyAllWindows()
    return


def all_color_maps_test():
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')
    cv_img = load_img_from_web(mtt.DOG, ack=False)
    cv_gray = cvt.bgr_to_gray(cv_img)
    cmaps = cvt.get_color_maps()  # max cmaps 22
    all_images = [cv_img, cv_gray]
    for color_map in cmaps:
        colored_img = cvt.add_color_map(cv_img=cv_gray, color_map=color_map)
        cvt.add_header(colored_img, header='idx {}'.format(color_map))
        all_images.append(colored_img)

    all_images.append(cvt.invert_image(cv_img))  # just to align the rows
    all_images.append(cvt.invert_image(cv_gray))
    for color_map in cmaps:  # do again with base inverted
        colored_img = cvt.add_color_map(cv_img=cv_gray, color_map=color_map, invert_base=True)
        cvt.add_header(colored_img, header='idx {}'.format(color_map))
        all_images.append(colored_img)

    cvt.display_open_cv_images(
        imgs=all_images,
        ms=BLOCK_MS_NORMAL,
        title='all_color_maps_test',
        loc='tl',
        resize=0.3,
        grid=(6, 8),  # 2 + 22 + 2 + 22 = 48 total with the originals (twice)
        header=None
    )

    cv2.destroyAllWindows()
    return


def draw_plus_test():
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')
    cv_img = load_img_from_web(mtt.DOG, ack=False)  # dog.jpg(159.92 KB) of shape (576, 768, 3)

    cvt.draw_plus(cv_img=cv_img, plus_center_xy=(150, 400), plus_color='lime', add_center_text=True)  # valid on the dog
    cvt.draw_plus(cv_img=cv_img, plus_center_xy=(200, 500), plus_color='red', add_center_text=True, text_color='b',
                  text_up=False)  # valid on the dog's tail
    cvt.draw_plus(cv_img=cv_img, plus_center_xy=(766, 400), plus_color='lime')  # overflow on x
    cvt.draw_plus(cv_img=cv_img, plus_center_xy=(0, 573), plus_color='lime')  # overflow on y
    cvt.draw_plus(cv_img=cv_img, plus_center_xy=(1000, 1000), plus_color='lime')  # completely outside of the image

    cvt.display_open_cv_image(
        img=cv_img,
        ms=BLOCK_MS_NORMAL,
        title='draw_plus_test',
        loc='tl',
        resize=None,
        header=None
    )
    cv2.destroyAllWindows()
    return


def draw_x_test():
    wu.get_function_name(ack=True, tabs=0)
    print('\tVisual test:')
    cv_img = load_img_from_web(mtt.DOG, ack=False)  # dog.jpg(159.92 KB) of shape (576, 768, 3)

    cvt.draw_x(cv_img=cv_img, x_center_xy=(150, 400), x_color='lime', add_center_text=True)  # valid on the dog
    cvt.draw_x(cv_img=cv_img, x_center_xy=(200, 500), x_color='red', add_center_text=True, text_color='b',
               text_up=False)  # valid on the dog's tail
    cvt.draw_x(cv_img=cv_img, x_center_xy=(766, 400), x_color='lime')  # overflow on x
    cvt.draw_x(cv_img=cv_img, x_center_xy=(0, 573), x_color='lime')  # overflow on y
    cvt.draw_x(cv_img=cv_img, x_center_xy=(1000, 1000), x_color='lime')  # completely outside of the image

    cvt.display_open_cv_image(
        img=cv_img,
        ms=BLOCK_MS_NORMAL,
        title='draw_plus_test',
        loc='tl',
        resize=None,
        header=None
    )
    cv2.destroyAllWindows()
    return


def get_roi_2dpoints_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img = load_img_from_web(mtt.DOG, ack=False)  # dog.jpg(159.92 KB) of shape (576, 768, 3)
    # cv_img = cvt.BGR_img_to_gray(cv_img)  # can do gray scale
    d2points = cvt.get_roi_2dpoints(cv_img, resize=1.6, mark_color='r', add_center_text=True,
                                    cursor_arrow=True)  # enlarge to 160%
    # d2points = cvt.get_roi_2dpoints(cv_img, resize=0.5, mark_color='r')  # reduce to 50%
    print(wu.to_str(d2points, '\td2points selected'))
    # cvt.display_open_cv_image(cv_img, title='x', ms=0, loc='tl', resize=None)  # check image not corrupted
    return


def get_named_roi_2dpoints_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img = load_img_from_web(mtt.DOG, ack=False)  # dog.jpg(159.92 KB) of shape (576, 768, 3)
    # cv_img = cvt.BGR_img_to_gray(cv_img)  # can do gray scale

    names_colors = OrderedDict([('port1_g', 'g'), ('port1_r', 'r'), ('port1_b', 'b')])

    colors_points = cvt.get_named_roi_2dpoints(
        cv_img, names_colors=names_colors, resize=1.6,
        add_center_text=True, cursor_arrow=True,
        loc='tl'
    )  # enlarge to 160%
    print(wu.to_str(colors_points, '\tcolors_points selected'))
    # cvt.display_open_cv_image(cv_img, title='x', ms=0, loc='tl', resize=None)  # check image not corrupted
    return


def get_named_roi_2dpoints_multi_image_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img = load_img_from_web(mtt.PERSON, ack=True)
    _, sketch_color = cvt.sketch_image(cv_bgr=cv_img)
    colored_img = cvt.add_color_map(cv_img=cv_img, color_map=cvt.get_color_maps(n=1)[0])
    colored_img2 = cvt.add_color_map(cv_img=cv_img, color_map=cvt.get_color_maps(n=1)[0])

    cvt.add_header(cv_img, header='port1')
    cvt.add_header(sketch_color, header='port2')
    cvt.add_header(colored_img, header='port3')
    cvt.add_header(colored_img2, header='port4')

    big_img = cvt.unpack_list_imgs_to_big_image(imgs=[cv_img, sketch_color, colored_img, colored_img2], grid=(2, 2))
    # cvt.display_open_cv_image(big_img)
    names_colors = OrderedDict([  # each image in it's turn
        ('port1_g', 'g'), ('port1_r', 'r'), ('port1_b', 'b'),
        ('port2_g', 'g'), ('port2_r', 'r'), ('port2_b', 'b'),
        ('port3_g', 'g'), ('port3_r', 'r'), ('port3_b', 'b'),
        ('port4_g', 'g'), ('port4_r', 'r'), ('port4_b', 'b'),
    ])

    # names_colors = OrderedDict([  # each color in it's turn
    #     ('port1_g', 'g'), ('port2_g', 'g'), ('port3_g', 'g'), ('port4_g', 'g'),
    #     ('port1_r', 'r'), ('port2_r', 'r'), ('port3_r', 'r'), ('port4_r', 'r'),
    #     ('port1_b', 'b'), ('port2_b', 'b'), ('port3_b', 'b'), ('port4_b', 'b')
    # ])

    named_points = cvt.get_named_roi_2dpoints(big_img, names_colors=names_colors, resize=0.8,
                                              add_center_text=True, cursor_arrow=True)  # enlarge to 160%

    print('\tnamed_points before unpacking:')
    for name, c_xy_dict in named_points.items():
        print('\t\t{}: c={}, xy={}'.format(name, c_xy_dict['c'], c_xy_dict['xy']))

    # on us to unpack correctly. since grid 2x2:
    # first image no offset
    # second image x offset: -width
    # third image y offset: -height
    # fourth image x offset -width and y offset -height
    h, w = cv_img.shape[:2]
    for name, c_xy_dict in named_points.items():
        x, y = c_xy_dict['xy']
        if 'port2' in name:
            c_xy_dict['xy'] = (x - w, y)
        elif 'port3' in name:
            c_xy_dict['xy'] = (x, y - h)
        elif 'port4' in name:
            c_xy_dict['xy'] = (x - w, y - h)

    # if marked the same roi on all images: should have around the same xy values
    print('\tnamed_points after unpacking:')
    for name, c_xy_dict in named_points.items():
        print('\t\t{}: c={}, xy={}'.format(name, c_xy_dict['c'], c_xy_dict['xy']))

    return


def is_bgr_and_3_cnl_img_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img = load_img_from_web(mtt.DOG, ack=True)  # dog.jpg(159.92 KB) of shape (576, 768, 3)
    print('\tis_bgr_or_rgb? {}'.format(cvt.is_bgr_or_rgb(cv_img)))
    print('\tis_gray? {}'.format(cvt.is_gray(cv_img)))
    print('\tconverting to gray and rechecking...')
    cv_img = cvt.bgr_to_gray(cv_img)
    print('\tis_bgr_or_rgb? {}'.format(cvt.is_bgr_or_rgb(cv_img)))
    print('\tis_gray? {}'.format(cvt.is_gray(cv_img)))
    return


def align_images_test():
    wu.get_function_name(ack=True, tabs=0)
    cv_img1 = load_img_from_web(mtt.DOG, ack=True)  # dog.jpg(159.92 KB) of shape (576, 768, 3)
    cv_img2 = load_img_from_web(mtt.PERSON, ack=True)
    cv_img3 = load_img_from_web(mtt.HORSES, ack=True)
    cv_img3 = cvt.bgr_to_gray(cv_img3, to_bgr_form=True)
    cv_img4 = load_img_from_web(mtt.FACES, ack=True)
    aligned_images = cvt.align_images(imgs=[cv_img1, cv_img2, cv_img3, cv_img4], pad_colors=['b'])
    cvt.display_open_cv_images(imgs=aligned_images, grid=(2, 2), resize=0.3, loc='tl', ms=BLOCK_MS_NORMAL,
                               title='align_images_test')
    aligned_images = cvt.align_images(imgs=[cv_img1, cv_img2, cv_img3, cv_img4], pad_colors=['g', 'w', 'purple'])
    cvt.display_open_cv_images(imgs=aligned_images, grid=(2, 2), resize=0.3, loc='tl', ms=BLOCK_MS_NORMAL,
                               title='align_images_test')
    cv2.destroyAllWindows()
    return


def test_all():
    print('{}{}:'.format('-' * 5, wu.get_base_file_and_function_name()))
    get_cv_version_test()
    imread_imwrite_test()
    list_to_cv_image_test()
    display_open_cv_image_test()
    display_open_cv_image_loop_test()
    resize_opencv_image_test()
    move_cv_img_x_y_test()
    move_cv_img_by_str_test()
    unpack_list_imgs_to_big_image_test()
    display_open_cv_images_test()
    display_open_cv_images_loop_test()
    gray_to_BGR_and_back_test()
    BGR_img_to_RGB_and_back_test()
    add_header_test()
    add_text_test()
    CameraWu_cv2_test()
    CameraWu_acapture_test()
    CameraWu_imutils_test()
    CameraWu_all_res_test()
    get_video_details_test()
    get_video_details_imutils_test()
    video_creator_mp4_test()
    video_creator_size_test()
    get_aspect_ratio_test()
    cuda_on_gpu_test()
    sketch_image_test()
    invert_image_test()
    add_color_map_test()
    all_color_maps_test()
    draw_plus_test()
    draw_x_test()
    # get_roi_2dpoints_test() # interactive test
    # get_named_roi_2dpoints_test()  # interactive test
    is_bgr_and_3_cnl_img_test()
    align_images_test()
    print('{}'.format('-' * 20))
    return
