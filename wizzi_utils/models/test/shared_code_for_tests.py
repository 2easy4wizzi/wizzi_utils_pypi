import numpy as np
import os
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
from wizzi_utils.pyplot import pyplot_tools as pyplt
from wizzi_utils.models.base_models import OdBaseModel
from wizzi_utils.models.base_models import PdBaseModel
# noinspection PyPackageRequirements
import cv2

# shared by more than one
DEBUG_DETECTIONS = False
# DEBUG_DETECTIONS = True  # TODO comment out
WITH_SUB_IMG = True
WITH_LABELS = True
CAM_FRAMES = 300
IMAGE_MS = 5000
VID_X_FRAMES_CV = 80
VID_X_FRAMES_TFL = 20

# object detection
WITH_TF = True
OD_OP = 1
OD_IM_TEST = True
OD_CAM_TEST = False
OD_VID_TEST = False

# pose detection
PD_OP = 1
PD_IM_TEST = True
PD_CAM_TEST = False
PD_VID_TEST = False

# tracking
TR_VID_TEST = True
TR_MODEL = 'CSRT'


# END ARGS

def od_run(
        model,  # od model
        cv_img: np.array,
        fps: mt.FPS
):
    fps.start()
    detections = model.detect_cv_img(cv_img, ack=DEBUG_DETECTIONS)
    print(detections)
    for i, detection_d in enumerate(detections):
        if i == 0:  # example how to put custom label
            detection_d['custom_label'] = 'cool_bicycle'

    fps.update(ack_progress=True, tabs=1, with_title=True)

    if WITH_TF:
        detections = model.add_traffic_light_to_detections(
            detections,
            traffic_light_p={
                'up': 0.2,
                'mid': 0.3,
                'down': 0.4
            },
            ack=DEBUG_DETECTIONS,
        )
    if WITH_SUB_IMG:
        detections = model.add_sub_sub_image_to_detection(
            detections,
            cv_img=cv_img,
            bbox_image_p={
                'x_left_start': 0.2,
                'x_left_end': 0.8,
                'y_top_start': 0,
                'y_top_end': 0.5,
            },
            ack=DEBUG_DETECTIONS,
        )
    model.draw_detections(
        detections,
        colors_d={
            'label_bbox': 'black',  # if draw_labels - text bg color
            'text': 'white',  # if draw_labels - text bg color
            'sub_image': 'blue',  # if draw_sub_image - sub image bbox color
            'default_bbox': 'red',  # bbox over the detection
            'person_bbox': 'black',  # custom color per class person
            'dog_bbox': 'lime',  # custom color per class dog
            'cat_bbox': 'magenta',  # custom color per class cat
        },
        cv_img=cv_img,
        draw_labels=WITH_LABELS,
        draw_tl_image=WITH_TF,
        draw_sub_image=WITH_SUB_IMG,
        header_tl={'text': 'wizzi_utils', 'x_offset': 0, 'text_color': 'aqua', 'with_rect': True, 'bg_color': 'black',
                   'bg_font_scale': 1},
        header_bl={'text': '{}-{}'.format(model.model_name, model.device), 'x_offset': 0, 'text_color': 'white',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
        header_tr={'text': '{:.2f} FPS'.format(fps.get_fps()), 'x_offset': 190, 'text_color': 'r', 'with_rect': False,
                   'bg_color': None, 'bg_font_scale': 1},
        header_br={'text': mt.get_time_stamp(format_s='%Y_%m_%d'), 'x_offset': 200, 'text_color': 'aqua',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
    )
    return


def pd_run(
        model,  # PD models
        cv_img: np.array,
        fps: mt.FPS
):
    """ shared by pose detection tests"""
    fps.start()
    detections = model.detect_cv_img(cv_img, ack=DEBUG_DETECTIONS)
    fps.update(ack_progress=True, tabs=1, with_title=True)

    if WITH_SUB_IMG:
        detections = model.add_sub_sub_image_to_detection(
            detections,
            cv_img=cv_img,
            bbox_image_p={
                'x_left_start': 0.2,
                'x_left_end': 0.8,
                'y_top_start': 0,
                'y_top_end': 0.5,
            },
            ack=DEBUG_DETECTIONS
        )
    model.draw_detections(
        detections,
        colors_d={
            'bbox_c': 'blue',
            'sub_image_c': 'black',
            'text_c': 'black',
        },
        cv_img=cv_img,
        draw_joints=True,
        draw_labels=WITH_LABELS,
        draw_edges=True,
        draw_bbox=WITH_SUB_IMG,
        draw_sub_image=WITH_SUB_IMG,
        header_tl={'text': 'wizzi_utils', 'x_offset': 0, 'text_color': 'aqua', 'with_rect': True, 'bg_color': 'black',
                   'bg_font_scale': 1},
        header_bl={'text': '{}-{}'.format(model.model_name, model.device), 'x_offset': 0, 'text_color': 'white',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
        header_tr={'text': '{:.2f} FPS'.format(fps.get_fps()), 'x_offset': 190, 'text_color': 'r', 'with_rect': False,
                   'bg_color': None, 'bg_font_scale': 1},
        header_br={'text': mt.get_time_stamp(format_s='%Y_%m_%d'), 'x_offset': 200, 'text_color': 'aqua',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
    )
    return


def od_or_pd_Model_image_test(
        cv_img: np.array,
        models: list,
        ms: int = cvtt.BLOCK_MS_NORMAL,
        dis_size: tuple = (640, 480),
        grid: tuple = (1, 1),
):
    mt.get_function_name(ack=True, tabs=0)
    cv_imgs_post = []

    for model in models:
        cv_img_clone = cv_img.copy()
        fps = mt.FPS(summary_title='{}'.format(model.model_name))
        if isinstance(model, OdBaseModel):
            od_run(model, cv_img_clone, fps=fps)
        elif isinstance(model, PdBaseModel):
            pd_run(model, cv_img_clone, fps=fps)
        cv_imgs_post.append(cv_img_clone)
    cvt.display_open_cv_images(
        cv_imgs_post,
        ms=ms,
        title='{}'.format(cv_img.shape),
        loc=pyplt.Location.CENTER_CENTER.value,
        grid=grid,
        resize=dis_size,
        header=None,
        save_path=None
    )
    cv2.destroyAllWindows()
    return


def _od_or_pd_Model_cam_test(
        cam: (cv2.VideoCapture, cvt.CameraWu),
        models: list,
        max_frames: int = CAM_FRAMES,
        work_every_x_frames: int = 1,
        dis_size: tuple = (640, 480),
        grid: tuple = (1, 1),
):
    if isinstance(cam, cv2.VideoCapture):
        img_w, img_h = cvt.get_dims_from_cap_cv2(cam)
    else:
        img_w, img_h = cvt.get_dims_from_cap_cv2(cam.cam)
    print('\tFrame dims {}x{}'.format(img_w, img_h))

    if len(models) == 1:
        mp4_out_dir = '{}/{}/{}'.format(mtt.VIDEOS_OUTPUTS, mt.get_function_name(), 'model')
        if not os.path.exists(mp4_out_dir):
            mt.create_dir(mp4_out_dir)
        out_fp = '{}/{}_detected.mp4'.format(mp4_out_dir, 'cam')
        out_dims = (img_w, img_h)
        mp4 = cvt.VideoCreator(
            out_full_path=out_fp,
            out_fps=20.0,
            out_dims=out_dims,
            codec='mp4v'
        )
        print(mp4)
    else:
        mp4 = None

    fps_list = [mt.FPS(last_k=10, summary_title='{}'.format(model.model_name)) for model in models]
    for i in range(max_frames):
        if isinstance(cam, cv2.VideoCapture):
            success, cv_img = cam.read()
        else:
            success, cv_img = cam.read_img()
        if i % work_every_x_frames != 0:  # s
            # do only x frames
            continue
        cv_imgs_post = []
        if success:
            for model, fps in zip(models, fps_list):
                cv_img_clone = cv_img.copy()
                if isinstance(model, OdBaseModel):
                    od_run(model, cv_img_clone, fps=fps)
                elif isinstance(model, PdBaseModel):
                    pd_run(model, cv_img_clone, fps=fps)
                cv_imgs_post.append(cv_img_clone)
                # if i == 0:  # first iter take much longer - measure from second iter
                #     fps.clear()
            if mp4 is not None:
                mp4.add_frame(cv_imgs_post[0])
        else:
            mt.exception_error(e='failed to grab frame {}'.format(i))
            continue
        k = cvt.display_open_cv_images(
            cv_imgs_post,
            ms=1,
            title='cam 0',
            loc=pyplt.Location.CENTER_CENTER.value,
            grid=grid,
            resize=dis_size,
            header='{}/{}'.format(i + 1, max_frames),
            save_path=None
        )
        if k == ord('q'):
            mt.exception_error('q was clicked. breaking loop')
            break
    for fps in fps_list:
        fps.finalize()
    if mp4 is not None:
        mp4.finalize()
    cv2.destroyAllWindows()
    return


def od_or_pd_Model_cam_test(
        models: list,
        max_frames: int = CAM_FRAMES,
        dis_size: tuple = (640, 480),
        grid: tuple = (1, 1),
):
    mt.get_function_name(ack=True, tabs=0)
    cam = cvt.CameraWu.open_camera(port=0, type_cam='cv2')

    if cam is not None:
        _od_or_pd_Model_cam_test(
            cam=cam,
            models=models,
            max_frames=max_frames,
            work_every_x_frames=1,
            dis_size=dis_size,
            grid=grid,
        )
    return


def od_or_pd_Model_video_test(
        models: list,
        vid_path: str,
        work_every_x_frames: int = 1,
        dis_size: tuple = (640, 480),
        grid: tuple = (1, 1),
):
    mt.get_function_name(ack=True, tabs=0)
    cam = cv2.VideoCapture(vid_path)
    if cam is not None:
        _od_or_pd_Model_cam_test(
            cam=cam,
            models=models,
            max_frames=cvt.get_frames_from_cap_cv2(cam),
            work_every_x_frames=work_every_x_frames,
            dis_size=dis_size,
            grid=grid,
        )
    return
