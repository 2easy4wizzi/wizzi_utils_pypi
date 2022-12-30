import numpy as np
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.models import models_configs as cfg
from wizzi_utils.models.base_models import PdBaseModel
# noinspection PyPackageRequirements
import cv2


class Cv2PdModel(PdBaseModel):

    def __init__(self,
                 save_load_dir: str,
                 model_name: str,
                 allowed_joint_names: list = None,
                 min_p_valid_joints: float = 0.3,
                 threshold: float = None,
                 in_dims: tuple = None,
                 scalefactor: float = None,
                 mean: tuple = None,
                 swapRB: bool = None,
                 crop: bool = None,
                 check_type: bool = True,
                 device: str = 'cpu'
                 ):
        """
        :param save_load_dir: where the model is saved (or will be if not exists)
        :param model_name: valid name in MODEL_CONF.keys()
        :param allowed_joint_names: joint_names to track from the joint_names of the model
        :param min_p_valid_joints:
            let x=min_p_valid_joints*len(allowed_joint_names)
            to keep a pose: we need max(x,2) joints found
        for all the following - if is None: take default value from MODELS_DNN_OBJECT_DETECTION_META_DATA['model_name']
        :param threshold: only detection above this threshold will be returned
        :param in_dims:
        :param scalefactor:
        :param mean:
        :param swapRB:
        :param crop:
        example:
        see:
        """
        super().__init__(save_load_dir=save_load_dir, model_name=model_name, allowed_joint_names=allowed_joint_names,
                         min_p_valid_joints=min_p_valid_joints)
        if check_type and not self.model_type_valid(self.model_cfg['model_type'], cfg.ModelType.PdCvNormal.value):
            exit(-1)
        self.pose_net = None
        if self.model_cfg['family'] == cfg.DnnFamily.Caffe.value:
            model_prototxt = "{}/{}.prototxt".format(self.local_path, self.model_name)
            model_caffe = "{}/{}.caffemodel".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_prototxt, url_dict=self.model_cfg['prototxt'])
            self._download_if_needed(local_path=model_caffe, url_dict=self.model_cfg['caffemodel'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_prototxt), mt.file_or_folder_size(model_caffe)]
            self.pose_net = cv2.dnn.readNetFromCaffe(prototxt=model_prototxt, caffeModel=model_caffe)

        elif self.model_cfg['family'] == cfg.DnnFamily.Darknet.value:
            model_cfg = "{}/{}.cfg".format(self.local_path, self.model_name)
            model_weights = "{}/{}.weights".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_cfg, url_dict=self.model_cfg['cfg'])
            self._download_if_needed(local_path=model_weights, url_dict=self.model_cfg['weights'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_cfg), mt.file_or_folder_size(model_weights)]
            self.pose_net = cv2.dnn.readNetFromDarknet(cfgFile=model_cfg, darknetModel=model_weights)

        elif self.model_cfg['family'] == cfg.DnnFamily.TF.value:
            model_pbtxt = "{}/{}.pbtxt".format(self.local_path, self.model_name)
            model_pb = "{}/{}.pb".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_pbtxt, url_dict=self.model_cfg['pbtxt'])
            self._download_if_needed(local_path=model_pb, url_dict=self.model_cfg['pb'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_pbtxt), mt.file_or_folder_size(model_pb)]
            self.pose_net = cv2.dnn.readNetFromTensorflow(model=model_pb, config=model_pbtxt)

        if self.pose_net is None:
            mt.exception_error('Failed to create network', real_exception=False)
            exit(-1)

        self.device = self.set_device(self.pose_net, device)

        self.pairs_indices = self.model_cfg['pairs_indices']

        if threshold is not None:
            self.model_cfg['threshold'] = threshold
        if in_dims is not None:
            self.model_cfg['in_dims'] = in_dims
        if scalefactor is not None:
            self.model_cfg['scalefactor'] = scalefactor
        if mean is not None:
            self.model_cfg['mean'] = mean
        if swapRB is not None:
            self.model_cfg['swapRB'] = swapRB
        if crop is not None:
            self.model_cfg['crop'] = crop

        return

    def to_string(self, tabs: int = 1) -> str:
        tabs_s = tabs * '\t'
        string = '{}{}'.format(tabs_s, mt.add_color(string='Cv2PdModel:', ops='underlined'))
        string += '\n\t{}name= {} (resources size: {})'.format(tabs_s, self.model_name, self.model_resources_sizes)
        string += '\n\t{}Running on {}'.format(tabs_s, self.device)
        string += '\n\t{}local_path={}'.format(tabs_s, self.local_path)
        string += '\n\t{}{}'.format(tabs_s, mt.to_str(self.allowed_joint_names, 'allowed_joint_names'))
        string += '\n{}'.format(mt.dict_as_table(self.model_cfg, title='conf', fp=6, ack=False, tabs=tabs + 1))
        return string

    def detect_cv_img(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: open cv image
        :param fp: float precision on the score precentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts. see extract_results()
        """
        inpBlob = cv2.dnn.blobFromImage(
            image=cv_img,
            scalefactor=self.model_cfg['scalefactor'],
            size=self.model_cfg['in_dims'],
            mean=self.model_cfg['mean'],
            swapRB=self.model_cfg['swapRB'],
            crop=self.model_cfg['crop'])
        self.pose_net.setInput(inpBlob)
        outputs = self.pose_net.forward()
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(outputs, 'outputs')))

        if len(outputs) > 0:
            outputs = outputs[0]

        detections = self.extract_results(
            outputs=outputs,
            cv_img=cv_img,
            fp=fp,
            ack=ack,
            tabs=tabs,
            img_title=img_title
        )
        return detections

    def extract_results(
            self,
            outputs: np.array,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param outputs: np array that contains the outputs for the cv_img
        :param cv_img: cv image
        :param fp: float precision on the score precentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts:
        dict is a detection of pose
            each has items:
                'joint_names_list':  - all joint names like in the config
                'joint_ids_list': - all joint ids like in the config
                'joint_x_y_list': - if joint found: it's x,y values
                'joint_z_list': - if joint found: it's z value
                'score_percentages_list': - if joint found: it's confidence in 0-100%,
        """

        if ack:
            title_suffix = '' if img_title is None else '{} '.format(img_title)
            title = '{} detection on image {}{}:'.format(self.model_name, title_suffix, cv_img.shape)
            print('{}{}'.format(tabs * '\t', title))
            print('{}Meta_data:'.format(tabs * '\t'))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(outputs, 'outputs')))
            print('{}Detections:'.format(tabs * '\t'))

        detections = []  # each detection is a set of joint
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        net_h, net_w = outputs.shape[1], outputs.shape[2]

        for pose_id in range(1):  # currently supports 1 pose
            detection_d = {
                'joint_names_list': [],
                'joint_ids_list': [],
                'joint_x_y_list': [],
                'score_percentages_list': [],
                'bbox': {}
            }
            valid_joints_found = 0  # max is len(self.allowed_joint_names)
            for joint_id, joint_name in self.joint_names.items():
                detection_d['joint_names_list'].append(joint_name)
                detection_d['joint_ids_list'].append(joint_id)
                probMap = outputs[joint_id, :, :]  # confidence map.
                minVal, score_frac, minLoc, point = cv2.minMaxLoc(probMap)  # Find global maxima of the probMap.
                # Scale the point to fit on the original image
                x = int((img_w * point[0]) / net_w)
                y = int((img_h * point[1]) / net_h)
                score_percentage = round(score_frac * 100, fp)

                if self.model_cfg['threshold'] <= score_frac <= 1.0 and joint_name in self.allowed_joint_names:
                    detection_d['joint_x_y_list'].append([x, y])
                    detection_d['score_percentages_list'].append(score_percentage)
                    valid_joints_found += 1
                else:
                    detection_d['joint_x_y_list'].append(self.NOT_FOUND_PAIR)
                    detection_d['score_percentages_list'].append(self.NOT_FOUND_VALUE)

                if ack:
                    d_msg = '{}\tpose {}: {}({}): xy={}, score=({}%)'
                    print(d_msg.format(tabs * '\t', pose_id, joint_name, joint_id, detection_d['joint_x_y_list'][-1],
                                       detection_d['score_percentages_list'][-1]))
            min_joints_found_ok = valid_joints_found >= self.min_p_valid_joints * len(self.allowed_joint_names)
            pose_valid = min_joints_found_ok and valid_joints_found >= 2
            if pose_valid:  # minimum 2 regardless to self.min_p_valid_joints(which could be zero)
                detection_d['bbox'] = PdBaseModel.get_bbox_dict(detection_d['joint_x_y_list'])
                if detection_d['bbox'] is not None:
                    if ack:
                        print('{}\tbbox:{}'.format(tabs * '\t', detection_d['bbox']))
                    detections.append(detection_d)
                else:
                    if ack:
                        print('{}\tPose {} was dumped. bbox is None'.format(tabs * '\t', pose_id))
            else:
                if ack:
                    min_needed = max(self.min_p_valid_joints * len(self.allowed_joint_names), 2)
                    msg = '{}\tPose {} was dumped. It had valid_joints_found={} (min needed {})'
                    print(msg.format(tabs * '\t', pose_id, valid_joints_found, min_needed))
        return detections

    @staticmethod
    def set_device(net: cv2.dnn, device: str = 'cpu') -> str:
        string = 'CPU'
        if device == "gpu":
            if cvt.cuda_on():
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                string = 'GPU'
            else:
                mt.exception_error('GPU requested but open_cv can\'t find cuda device')
                net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        else:  # device == "cpu":
            net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        return string


class Cv2PdModelCocoMultiPoses(Cv2PdModel):
    def __init__(self,
                 save_load_dir: str,
                 model_name: str,
                 allowed_joint_names: list = None,
                 min_p_valid_joints: float = 0.3,
                 threshold: float = None,
                 in_dims: tuple = None,
                 scalefactor: float = None,
                 mean: tuple = None,
                 swapRB: bool = None,
                 crop: bool = None,
                 check_type: bool = True,
                 device: str = 'cpu'
                 ):
        """
        :param save_load_dir: where the model is saved (or will be if not exists)
        :param model_name: valid name in MODEL_CONF.keys()
        for all the following - if is None: take default value from MODELS_DNN_OBJECT_DETECTION_META_DATA['model_name']
        :param threshold: only detection above this threshold will be returned
        :param in_dims:
        :param scalefactor:
        :param mean:
        :param swapRB:
        :param crop:
        example:
        see:
        """
        # inWidth = int((inHeight / frameHeight) * frameWidth)
        super().__init__(
            save_load_dir=save_load_dir, model_name=model_name, allowed_joint_names=allowed_joint_names,
            min_p_valid_joints=min_p_valid_joints, threshold=threshold, in_dims=in_dims, scalefactor=scalefactor,
            mean=mean, swapRB=swapRB, crop=crop, check_type=False, device=device
        )
        if check_type and not self.model_type_valid(self.model_cfg['model_type'], cfg.ModelType.PdCvCocoMulti.value):
            exit(-1)

        self.pose_pairs_indices = self.pairs_indices
        self.pafs_pairs_indices = self.model_cfg['pafs_indices']
        return

    @staticmethod
    def extract_joint_locations(probMap: np.array, threshold: float, seq_id_start: int) -> (dict, int):
        """
        :param probMap: joint probMap of a joint in size (cv_img_width, cv_img_height)
        :param threshold:
        :param seq_id_start: used to give an id for each detection.
        :return: list of dicts: per joint detection in every pose - see self.joint_names
        """
        mapSmooth = cv2.GaussianBlur(src=probMap, ksize=(3, 3), sigmaX=0, dst=0)
        mapMask = np.uint8(mapSmooth > threshold)
        joint_data = {}

        # find the blobs
        contours, hierarchy = cv2.findContours(
            image=mapMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        # for each blob find the maxima
        for i, cnt in enumerate(contours):
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(img=blobMask, points=cnt, color=1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            confidence = probMap[maxLoc[1], maxLoc[0]]  # float
            joint_data[seq_id_start + i] = {
                'xy': list(maxLoc),  # maxLoc is a xy - 2d list
                'score_frac': confidence
            }
        seq_id_end = seq_id_start + len(joint_data)
        return joint_data, seq_id_end

    def get_all_joints(self, outputs: np.array, cv_img: np.array, threshold: float) -> dict:
        """
        :param outputs: detection output
        :param cv_img: original image
        :param threshold: threshold
        :return:
        * seq_id is just a sequential running id to each joint found.
            e.g. we have 2 noses, 2 necks ...
                so nose 1 seq_id = 0
                so nose 2 seq_id = 1
                so neck 1 seq_id = 2
                and so on

        dict of joint_id to joint_datum_dict
            joint_datum_dict is a dict of seq_id to joint_data_dict
                joint_data_dict has 'xy' and 'score_frac'
        """
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        all_joints_datum = {}
        seq_id = 0
        for joint_id, joint_name in self.joint_names.items():
            probMap = outputs[joint_id, :, :]  # probability map for this joint - multi detections
            probMap = cv2.resize(probMap, (img_w, img_h))
            # print(mt.to_str(probMap, '\t\tprobMap'))
            # cvt.display_open_cv_image(probMap, ms=0, title='probMap')  # you can actual image show this map
            joint_datum_dict, seq_id = Cv2PdModelCocoMultiPoses.extract_joint_locations(probMap, threshold, seq_id)
            all_joints_datum[joint_id] = joint_datum_dict

        # uncomment to see output
        # detected all joints but don't know which belongs to which person
        # print(mt.to_str(all_joints_datum, '\tall_joints_datum'))
        # for joint_id, joint_datum_dict in all_joints_datum.items():
        #     print('{}({})'.format(self.joint_names[joint_id], joint_id))
        #     for seq_id, joint_data_dict in joint_datum_dict.items():
        #         print('\tseq_id {}: {}'.format(seq_id, joint_data_dict))

        # uncomment to draw all joints
        # for joint_id, joint_datum_dict in all_joints_datum.items():  # all_joints_datum is a list of lists
        #     for seq_id, joint_data_dict in joint_datum_dict.items():
        #         cv2.circle(cv_img, joint_data_dict['xy'], 2, pyplt.get_BGR_color('blue'), -1, cv2.LINE_AA)
        # cvt.display_open_cv_image(cv_img, ms=1, title='cv_img - all joints')
        return all_joints_datum

    def join_joints_of_same_pose(self, output: np.array, all_joints_datum: dict, cv_img: np.array) -> dict:
        """
        :param output:
        :param all_joints_datum: see extract_joint_locations() output
            e.g. for joint_id == 0 (the nose), we have a list of noses found on image
        :param cv_img:
        :return:
            Find valid connections between the different joints of a all persons present:
            for pose_pair_indices in self.pose_pairs_indices: (key="pose_pair_indices[0]_pose_pair_indices[1]")
                valid_pairs_dict[key] = valid_pair_list
            e.g. go over each pose pair indices (the first is [1,0]: neck-nose):
                take the list of noses detected and the list of necks from all_joints_datum[0] and [1]
                now figure who is most likely to be connected to who
        """
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        valid_pairs_dict = {}
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR and see which seq_idA is connected to seq_idB
        for k, (pose_pair_indices, pafs_pair_indices) in enumerate(
                zip(self.pose_pairs_indices, self.pafs_pairs_indices)):
            joint_id_1, joint_id_2 = pose_pair_indices
            # Find the key points for the first and second limb
            candidatesA = all_joints_datum[joint_id_1]
            candidatesB = all_joints_datum[joint_id_2]
            key = '{}_{}'.format(joint_id_1, joint_id_2)
            paf_idx_1, paf_idx_2 = pafs_pair_indices

            # A->B constitute a limb
            pafA = output[paf_idx_1]
            pafB = output[paf_idx_2]
            pafA = cv2.resize(pafA, (img_w, img_h))
            pafB = cv2.resize(pafB, (img_w, img_h))

            # If key points for the joint-pair is detected
            # check every joint in candidatesA with every joint in candidatesB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

            if len(candidatesA) != 0 and len(candidatesB) != 0:
                valid_pairs = []
                for seq_idA, candidateA in candidatesA.items():
                    xy_i = np.array(candidateA['xy'])
                    best_seq_idB = -1
                    maxScore = -1
                    for seq_idB, candidateB in candidatesB.items():
                        xy_j = np.array(candidateB['xy'])
                        d_ij = xy_j - xy_i  # Find d_ij
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                            interp_coord = list(zip(  # Find p(u)
                                np.linspace(xy_i[0], xy_j[0], num=n_interp_samples),
                                np.linspace(xy_i[1], xy_j[1], num=n_interp_samples)
                            ))

                            paf_interp = []
                            for xy in interp_coord:  # Find L(p(u))
                                xy_r = np.round(xy).astype(np.int)
                                paf_interp.append([pafA[xy_r[1], xy_r[0]], pafB[xy_r[1], xy_r[0]]])

                            paf_scores = np.dot(paf_interp, d_ij)  # Find E
                            avg_paf_score = sum(paf_scores) / len(paf_scores)

                            # Check if the connection is valid
                            # If the fraction of interpolated vectors aligned with PAF
                            # is higher then threshold -> Valid Pair
                            if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                                if avg_paf_score > maxScore:
                                    maxScore = avg_paf_score
                                    best_seq_idB = seq_idB
                    # Append the connection to the list
                    if best_seq_idB != -1:
                        valid_pairs.append([seq_idA, best_seq_idB])

                # Append the detected connections to the global list
                valid_pairs_dict[key] = valid_pairs
            else:  # If no valid pairs are detected - save joints_pair_index
                # mt.exception_error('No Connection : k = {}'.format(k), real_exception=False)
                valid_pairs_dict[key] = []
        # uncomment for output
        # print(mt.to_str(valid_pairs_dict, '\tvalid_pairs_dict', chars='all'))
        # for pose_pair_indices, valid_pairs in valid_pairs_dict.items():
        #     p1, p2 = pose_pair_indices.split('_')
        #     p1, p2 = int(p1), int(p2)
        #     print('\t\tconnection of {}({}) to {}({})'.format(self.joint_names[p1], p1, self.joint_names[p2], p2))
        #     print('\t\t\tseq_ids list {}'.format(valid_pairs))
        return valid_pairs_dict

    def separate_poses(self, valid_pairs_dict: dict):
        """
        :param valid_pairs_dict:
        :return:
        using the valid_pairs_dict - separate poses
        e.g we have 2 poses (not mandatory all joints found on both poses)
        first pose_pairs_indices is 1,2 which is neck to RShoulder
        so an example to a valid pair in valid_pairs_dict['1_2']:
            connection of Neck(1) to RShoulder(2)
                seq_ids list [[1, 3], [2, 4]]
            all_joints_datum[joint_id='1'][seq_id='1'] is the 1st Neck
            all_joints_datum[joint_id='1'][seq_id='2'] is the 2nd Neck
            all_joints_datum[joint_id='2'][seq_id='3'] is the 1st RShoulder
            all_joints_datum[joint_id='2'][seq_id='4'] is the 2nd RShoulder

        """
        poses = []

        for pose_pair in self.pose_pairs_indices:
            poseAid, poseBid = pose_pair
            key = '{}_{}'.format(poseAid, poseBid)
            valid_pairs = valid_pairs_dict[key]  # x seq_ids of connected joints for this pose_pair (e.g. neck to nose)

            if len(valid_pairs) > 0:
                valid_pairs = np.array(valid_pairs)
                seq_ids1 = valid_pairs[:, 0]  # take first joint seq_ids
                seq_ids2 = valid_pairs[:, 1]  # take second joint seq_ids

                for seq_id1, seq_id2 in zip(seq_ids1, seq_ids2):
                    person_idx = -1
                    for j, pose in enumerate(poses):
                        if pose[poseAid] == seq_id1:
                            person_idx = j
                            break
                    if person_idx != -1:
                        poses[person_idx][poseBid] = seq_id2

                    # new person found
                    elif person_idx == -1:
                        row = -1 * np.ones(len(self.joint_names))
                        row[poseAid] = seq_id1
                        row[poseBid] = seq_id2
                        poses.append(row)
        poses = np.array(poses, dtype=np.int)
        return poses

    def extract_results(
            self,
            outputs: np.array,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param outputs: np array that contains the outputs for the cv_img
        :param cv_img: cv image
        :param fp: float precision on the score precentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts:
        dict is a detection of pose
            each has items:
                'joint_names_list':  - all joint names like in the config
                'joint_ids_list': - all joint ids like in the config
                'joint_x_y_list': - if joint found: it's x,y values
                'joint_z_list': - if joint found: it's z value
                'score_percentages_list': - if joint found: it's confidence in 0-100%,
        """

        if ack:
            title_suffix = '' if img_title is None else '{} '.format(img_title)
            title = '{} detection on image {}{}:'.format(self.model_name, title_suffix, cv_img.shape)
            print('{}{}'.format(tabs * '\t', title))
            print('{}Meta_data:'.format(tabs * '\t'))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(outputs, 'outputs')))

        all_joints_datum = self.get_all_joints(outputs, cv_img, self.model_cfg['threshold'])
        valid_pairs_dict = self.join_joints_of_same_pose(outputs, all_joints_datum, cv_img)
        poses = self.separate_poses(valid_pairs_dict)
        if ack:
            print('{}\t{}'.format(tabs * '\t', mt.to_str(all_joints_datum, 'all_joints_datum', chars=300)))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(valid_pairs_dict, 'valid_pairs_dict')))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(poses, 'poses')))
            print('{}Detections:'.format(tabs * '\t'))

        # cvt.display_open_cv_image(cv_img, ms=1, title='cv_img - all joints')

        detections = []  # each detection is a set of joints
        for pose_id, pose in enumerate(poses):
            if ack:
                print('{}\t{}'.format(tabs * '\t', mt.to_str(pose, 'pose {}'.format(pose_id))))
            detection_d = {
                'joint_names_list': [],
                'joint_ids_list': [],
                'joint_x_y_list': [],
                'score_percentages_list': [],
                'bbox': {}
            }
            valid_joints_found = 0  # max is len(self.allowed_joint_names)
            for joint_id, joint_name in self.joint_names.items():
                detection_d['joint_names_list'].append(joint_name)
                detection_d['joint_ids_list'].append(joint_id)
                seq_id = pose[joint_id]

                if seq_id != self.NOT_FOUND_VALUE:
                    joint_d = all_joints_datum[joint_id][seq_id]
                    x, y = joint_d['xy']
                    score_frac = joint_d['score_frac']

                    if self.model_cfg['threshold'] <= score_frac <= 1.0 and joint_name in self.allowed_joint_names:
                        score_percentage = round(score_frac * 100, fp)
                        detection_d['joint_x_y_list'].append([x, y])
                        detection_d['score_percentages_list'].append(score_percentage)
                        valid_joints_found += 1
                    else:
                        detection_d['joint_x_y_list'].append(self.NOT_FOUND_PAIR)
                        detection_d['score_percentages_list'].append(self.NOT_FOUND_VALUE)
                else:
                    detection_d['joint_x_y_list'].append(self.NOT_FOUND_PAIR)
                    detection_d['score_percentages_list'].append(self.NOT_FOUND_VALUE)
                if ack:
                    d_msg = '{}\t\tpose {}: {}({}): xy={}, score=({}%)'
                    print(d_msg.format(tabs * '\t', pose_id, joint_name, joint_id, detection_d['joint_x_y_list'][-1],
                                       detection_d['score_percentages_list'][-1]))

            min_joints_found_ok = valid_joints_found >= self.min_p_valid_joints * len(self.allowed_joint_names)
            pose_valid = min_joints_found_ok and valid_joints_found >= 2
            if pose_valid:  # minimum 2 regardless to self.min_p_valid_joints(which could be zero)
                detection_d['bbox'] = PdBaseModel.get_bbox_dict(detection_d['joint_x_y_list'])
                if detection_d['bbox'] is not None:
                    if ack:
                        print('{}\tbbox:{}'.format(tabs * '\t', detection_d['bbox']))
                    detections.append(detection_d)
                else:
                    if ack:
                        print('{}\tPose {} was dumped. bbox is None'.format(tabs * '\t', pose_id))
            else:
                if ack:
                    min_needed = max(self.min_p_valid_joints * len(self.allowed_joint_names), 2)
                    msg = '{}\tPose {} was dumped. It had valid_joints_found={} (min needed {})'
                    print(msg.format(tabs * '\t', pose_id, valid_joints_found, min_needed))
        return detections
