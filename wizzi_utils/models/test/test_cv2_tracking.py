from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
from wizzi_utils.pyplot import pyplot_tools as pyplt
import numpy as np
# noinspection PyPackageRequirements
import cv2
from wizzi_utils.models.cv2_models import tracking as tr
from wizzi_utils.models.test.shared_code_for_tests import TR_MODEL, TR_VID_TEST, DEBUG_DETECTIONS, WITH_LABELS


def tr_run(
        model: tr.Cv2Tracker,
        cv_img: np.array,
        frame_idx: int,
        fps: mt.FPS
):
    fps.start()
    detections = model.track_frame(cv_img, ack=DEBUG_DETECTIONS, tabs=0, frame_idx=frame_idx)
    fps.update(ack_progress=False, tabs=1, with_title=True)

    model.draw_detections(
        detections,
        cv_img=cv_img,
        colors_d={
            'default_bbox_color': 'lime',
            'bbox_color_1': 'red',  # custom bbox color for tracker id == 1
        },
        draw_labels=WITH_LABELS,
        tracks_dict={
            'default_tracks_color': 'blue',
            'len_tracks': 100,
            'radius': 1,
            'thickness': -1,
            'tracks_color_1': 'yellow'  # custom tracks color for tracker id == 1
        },
        # header_tl={'text': 'wizzi_utils', 'x_offset': 0, 'text_color': 'aqua', 'with_rect': True, 'bg_color': 'black',
        #            'bg_font_scale': 1},
        header_bl={'text': '{}'.format(model.tracker_type), 'x_offset': 0, 'text_color': 'white',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
        header_tr={'text': '{:.2f} FPS'.format(fps.get_fps()), 'x_offset': 190, 'text_color': 'r', 'with_rect': False,
                   'bg_color': None, 'bg_font_scale': 1},
        header_br={'text': mt.get_time_stamp(format_s='%Y_%m_%d'), 'x_offset': 200, 'text_color': 'aqua',
                   'with_rect': True, 'bg_color': 'black', 'bg_font_scale': 1},
    )

    return


def tr_Model_video_test(
        model_name: str,
        vid_path: str,
        work_every_x_frames: int = 1,
        in_dims_w: int = None
):
    mt.get_function_name(ack=True, tabs=0)

    cam = cv2.VideoCapture(vid_path)
    if cam is not None:
        img_w, img_h = cvt.get_dims_from_cap_cv2(cam)
        print('\tFrame dims {}x{}'.format(img_w, img_h))
        if in_dims_w is None:
            in_dims = (img_w, img_h)  # work on orig image
        else:
            # work on w=in_dims_w and h derived from w
            in_dims = cvt.get_aspect_ratio_h(img_w=img_w, img_h=img_h, new_w=in_dims_w)

        model = tr.Cv2Tracker(model_obj=tr.Trackers[model_name])
        print(model.to_string())

        fps = mt.FPS(last_k=10, summary_title='{}'.format(model.tracker_type))
        max_frames = cvt.get_frames_from_cap_cv2(cam)
        title = 'cv tracker'
        for i in range(max_frames):
            success, cv_img = cam.read()
            if i % work_every_x_frames != 0:  # s
                # do only x frames
                continue
            if success:
                cv_img = cv2.resize(cv_img, in_dims)
                tr_run(model, cv_img, frame_idx=i, fps=fps)
                # if i == 0:  # first iter take much longer - measure from second iter
                #     fps.clear()
                k = cvt.display_open_cv_image(
                    cv_img,
                    ms=1,
                    title=title,
                    loc=pyplt.Location.CENTER_CENTER.value,
                    resize=None,
                    # resize=(1280, 720),
                    header='{}/{}'.format(i + 1, max_frames),
                    save_path=None
                )
                if k == ord('q'):
                    mt.exception_error('q was clicked. breaking loop')
                    break
                if k == ord('s'):
                    if isinstance(model, tr.Cv2Tracker):
                        model.add_obj_bbox_roi(title, cv_img, ack=DEBUG_DETECTIONS, frame_idx=i)

                if vid_path.endswith('woman_yoga.mp4'):
                    # ADD manually object that i know were there to save you dealing with ROI
                    if i == 14:
                        xywh = (175, 66, 194, 219)  # woman in original image space
                        bbox = model.change_dims(xywh=xywh, old_dims=(img_w, img_h), new_dims=in_dims)
                        bbox = model.tuple_to_int(bbox)
                        model.add_obj_bbox_manual(cv_img, xywh=bbox, frame_idx=i, name='woman', ack=DEBUG_DETECTIONS)
                    # if i == 40:  # checking if active == False works fine
                    #     model.get_tracker_dict(tracker_id=0)['active'] = False
                    if i == 36:
                        xywh = (241, 285, 68, 70)  # cat in original image space
                        bbox = model.change_dims(xywh=xywh, old_dims=(img_w, img_h), new_dims=in_dims)
                        bbox = model.tuple_to_int(bbox)
                        model.add_obj_bbox_manual(cv_img, xywh=bbox, frame_idx=i, name='cat', ack=DEBUG_DETECTIONS)
            else:
                mt.exception_error(e='failed to grab frame {}'.format(i))
                continue
        fps.finalize()
        cv2.destroyAllWindows()
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    if TR_VID_TEST:
        vid_path = cvtt.get_vid_from_web(name=mtt.WOMAN_YOGA)
        tr_Model_video_test(
            model_name=TR_MODEL,  # slower but better
            vid_path=vid_path,
            work_every_x_frames=1,
            # in_dims_w=500,  # if None - work on original image size
        )
    print('{}'.format('-' * 20))
    return
