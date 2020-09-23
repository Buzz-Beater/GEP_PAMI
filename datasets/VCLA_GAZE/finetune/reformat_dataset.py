"""
Created on 10/18/18

@author: Baoxiong Jia

Description:

"""

import os
import time
import glob
import re
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import pickle
import shutil

import sys
sys.path.append('/mnt/hdd/home/baoxiong/Projects/TPAMI2019/src')

import datasets.VCLA_GAZE.vcla_gaze_config as vcla_gaze_config
from datasets.VCLA_GAZE.metadata import VCLA_METADATA
import utils.logutils as utils
metadata = VCLA_METADATA()


def load_skeleton(path, index):
    skeletons = []
    for id in index:
        id = id[0]
        skeleton_file = glob.glob(os.path.join(path, 'skeleton_{0:05}_*.txt'.format(id)))

        if len(skeleton_file) == 0:
            raise ValueError('{}/{}'.format(path, id))
        with open(skeleton_file[0], 'r') as f:
            skeleton_data = np.array([list(map(float, info.rstrip('\r\n').split()))
                                        for idx, info in enumerate(f.readlines()) if idx in range(1, 26)])
        skeleton_3d = skeleton_data[:, 1 : 4]
        skeleton_depth = skeleton_data[:, 5 : 7]
        skeleton_gui = skeleton_data[:, 7 : 9]
        skeletons.append(np.hstack((skeleton_3d, skeleton_depth, skeleton_gui)))
    skeletons = np.array(skeletons)
    return skeletons

# Deprecated
def align_skeleton(skeleton, mean_skeleton):
    mean_skeleton_3d = mean_skeleton[:, : 3]
    mean_skeleton_depth = mean_skeleton[:, 3 : 5]
    mean_skeleton_gui = mean_skeleton[:, 5 : 7]

    skeleton_3d = skeleton[:, : 3]
    skeleton_depth = skeleton[:, 3 : 5]
    skeleton_gui = skeleton[:, 5 : 7]

    aligned_skeleton_3d = utils.transform(skeleton_3d, mean_skeleton_3d, dims=3)
    aligned_skeleton_depth = utils.transform(skeleton_depth, mean_skeleton_depth, dims=2)
    aligned_skeleton_gui = utils.transform(skeleton_gui, mean_skeleton_gui, dims=2)

    return np.hstack((aligned_skeleton_3d, aligned_skeleton_depth, aligned_skeleton_gui))

def load_info_all(paths, categories, video_ids):
    frame_ids_all = dict()
    annotations_all = dict()
    obj_bboxs_all = dict()
    obj_classes_all = dict()
    skeletons_all = dict()
    image_ids_all = dict()
    for cat_idx, category in enumerate(tqdm(categories, desc='Category Loop')):
        tqdm.write('--> Loading Category {}'.format(category))
        for video_id in tqdm(video_ids[cat_idx], desc='Video Loop', leave=False):
            tqdm.write('------> Loading Video {}'.format(video_id))
            sequence_id = category + '$' + video_id
            # Get all available image list
            video_path = os.path.join(paths.img_root, category, video_id, 'TPV') # Using Third Person View
            image_list = sorted(glob.glob(os.path.join(video_path, 'raw_depth*')))
            pattern = video_path + '/raw_depth_([0-9]+)_[0-9]+.png'
            image_ids = [int(re.findall(pattern, img_name)[0]) for img_name in image_list]

            anno_file = os.path.join(paths.anno_root, category, video_id + '.txt')
            bbox_file = os.path.join(paths.bbox_root, category, video_id + '_ObjBox.mat')
            frame_id_file = os.path.join(paths.label_root, 'frame_id', category, video_id + '_FrmID.mat')

            # Solve name inconsistent between image files, annotation files and object bounding box files
            # All renamed according to the image folder names
            try:
                with open(anno_file, 'r') as f:
                    annotations = f.readlines()
            except:
                pref_index = video_id[:3]  # Getting the id, e.g. SXX
                anno_file_name = glob.glob(
                    os.path.join(paths.data_root, 'labels', 'annotations', category, pref_index + '*'))
                dst_name = os.path.join(paths.data_root, 'labels', 'annotations', category, video_id + '.txt')
                os.rename(anno_file_name[0], dst_name)
            try:
                frame_id_mat = sio.loadmat(frame_id_file)
                frame_ids = frame_id_mat['FrmID']
            except:
                pref_index = video_id[:3]
                frmid_file_name = glob.glob(os.path.join(paths.label_root, 'frame_id', category, pref_index + '*'))
                frmid_dst_name = frame_id_file
                os.rename(frmid_file_name[0], frmid_dst_name)

            try:
                obj_bbox_mat = sio.loadmat(bbox_file)
                obj_bboxs = obj_bbox_mat['ObjBox']
                obj_classes = [obj_bbox_mat['ObjCls'][c_idx, 0][0] for c_idx in range(len(obj_bbox_mat['ObjCls']))]
            except:
                pref_index = video_id[:3]
                bbox_file_name = glob.glob(os.path.join(paths.bbox_root, category, pref_index + '*'))
                bbox_dst_name = bbox_file
                os.rename(bbox_file_name[0], bbox_dst_name)

            skeletons = load_skeleton(video_path, frame_ids)

            image_ids_all[sequence_id] = image_ids
            frame_ids_all[sequence_id] = frame_ids
            annotations_all[sequence_id] = annotations
            obj_bboxs_all[sequence_id] = obj_bboxs
            obj_classes_all[sequence_id] = obj_classes
            skeletons_all[sequence_id] = skeletons

    return image_ids_all, frame_ids_all, annotations_all, obj_bboxs_all, obj_classes_all, skeletons_all


def reformat_data(paths, option='Segment', verbose=False):
    """
    Reformat data annotations into frame-vectors, dump metadata files
    :param paths: paths to the dataset
    :return: None
    """
    anno_path = paths.anno_root
    img_path = paths.img_root
    corpus_path = os.path.join(paths.tmp_root, 'corpus')
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    else:
        shutil.rmtree(corpus_path)
        os.makedirs(corpus_path)

    categories = os.listdir(img_path)
    video_ids = [os.listdir(os.path.join(img_path, category)) for category in categories]
    if verbose:
        categories = ['c14_open_door']        # testing setting
        video_ids = [['S53_V10_T20161228_034010_00_FHXL2_PD9thFloor_G05']]
    anno_ids = [os.listdir(os.path.join(anno_path, category)) for category in categories]
    data_list = {}
    available_sequence_ids = []

    image_ids_all, frame_ids_all, annotations_all, obj_bboxs_all, obj_classes_all, skeletons_all \
                                                                        = load_info_all(paths, categories, video_ids)
    total_count = 0
    valid_count = 0

    # clean annotation
    for cat_idx, category in enumerate(tqdm(categories, desc='Category Loop')):
        tqdm.write('==> Processing Category {}'.format(category))
        # check annotation files amount
        assert (len(anno_ids) == len(video_ids))
        for video_id in tqdm(video_ids[cat_idx], desc='Video Loop', leave=False):
            total_count += 1
            tqdm.write('======> Processing Video{}'.format(video_id))
            sequence_id = category + '$' + video_id

            frame_ids = frame_ids_all[sequence_id]
            annotations = annotations_all[sequence_id]
            obj_bboxs = obj_bboxs_all[sequence_id]
            obj_classes = obj_classes_all[sequence_id]
            skeletons = skeletons_all[sequence_id]
            image_ids = image_ids_all[sequence_id]

            # Get existing image id list for checking the correctness of the annotation
            contents = [annotation.rstrip('\n').split(',') for annotation in annotations]
            start = -1
            end = -1
            for con_idx, content in enumerate(contents):
                start_frame = int(content[0])
                end_frame = int(content[1])
                if con_idx == 0:
                    start = start_frame
                if con_idx == len(contents) - 1:
                    end = end_frame
            # check if frame index matches index
            assert (abs(len(image_ids) - end + start - 1) <= 5)
            assert(abs(start - image_ids[0]) <= 5)
            assert(abs(end - image_ids[-1]) <= 5)

            # truncate frames according to available frame ids
            max_frmid = max(frame_ids)[0]
            min_frmid = min(frame_ids)[0]
            start_f = max(min_frmid, start)
            end_f = min(max_frmid, end)

            # remove videos that have skip inside
            if len(frame_ids) != max_frmid - min_frmid + 1:
                continue

            # reformatting info and dumping
            length = end_f - start_f + 1
            activity_list = []
            activity_mat = np.zeros((length))
            object_mat = np.zeros((length, metadata.MAXIMUM_OBJ_VIDEO, len(metadata.objects) + 4))
            # default to null object
            object_mat[:, :, 0] = 1
            affordance_mat = np.zeros((length, metadata.MAXIMUM_OBJ_VIDEO))
            # defualt to null affordance
            affordance_mat[:, 0] = 1
            skeleton_mat = np.zeros((length, len(skeletons[0]), len(skeletons[0][0])))
            segment = list()
            for con_idx, content in enumerate(contents):
                start_frame = int(content[0])
                end_frame = int(content[1])
                # cut the video to available frames
                if end_frame < min_frmid:
                    continue
                if start_frame > max_frmid:
                    continue
                if start_frame < min_frmid:
                    start_frame = min_frmid
                if end_frame > max_frmid:
                    end_frame = max_frmid
                activity = content[2]
                activity_list.append(activity)
                activity_mat[start_frame - start_f : end_frame - start_f + 1] = metadata.subactivity_index[activity]
                segment.append((start_frame, end_frame))
                for pair_idx in range(int(len(content) / 2 )- 1):
                    object = content[3 + pair_idx * 2]
                    object_rel_idx = obj_classes.index(object)
                    # Object bounding box index starts from min_frmid
                    object_bbox = obj_bboxs[start_frame - min_frmid : end_frame - min_frmid + 1,  \
                                                                object_rel_idx * 4 : (object_rel_idx + 1) * 4]
                    affordance = content[4 + pair_idx * 2]

                    # Check if its a valid bounding box, there are invalid bounding boxes
                    for frm_idx, frame in enumerate(range(start_frame, end_frame + 1)):
                        bbox = object_bbox[frm_idx]
                        upper_left_height = bbox[0]
                        upper_left_width = bbox[1]
                        bbox_height = bbox[2]
                        bbox_width = bbox[3]
                        # Annotation index starts from start_frame
                        object_vec = np.zeros(len(metadata.objects))
                        # Remove invalid bounding boxes
                        if not ((upper_left_height >= 0 and upper_left_height <= utils.rgb_height) and \
                                (upper_left_width >= 0 and upper_left_width <= utils.rgb_width) and \
                                (bbox_height >= 0 and bbox_height <= utils.rgb_height) and \
                                (bbox_width >= 0 and bbox_width <= utils.rgb_width)):
                            bbox = np.array([0., 0., 0., 0.], dtype=np.float)
                            object_name = 'null'
                            affordance_name = 'null'
                        else:
                            object_name = object
                            affordance_name = affordance
                        object_vec[metadata.object_index[object_name]] = 1
                        object_mat[frame - start_f, pair_idx, :] = np.hstack((object_vec, bbox))
                        affordance_mat[frame - start_f, pair_idx] = metadata.affordance_index[affordance_name]
                    skeleton_mat[start_frame - start_f : end_frame - start_f + 1, :, :] = \
                                            skeletons[start_frame - min_frmid : end_frame - min_frmid + 1, :, :]

            if option == 'Segment':
                valid_count += 1
                available_sequence_ids.append(sequence_id)
                data_list[sequence_id] = {'index': np.arange(start_f, end_f + 1), 'object_num': len(obj_classes),
                                          'activity_mat': activity_mat, 'affordance_mat': affordance_mat,
                                          'object_mat': object_mat, 'skeleton_mat': skeleton_mat, 'segment': segment}
            else:
                for frame_id in range(start_f, end_f + 1):
                    sequence_img_id = sequence_id + '$' + str(frame_id)
                    available_sequence_ids.append(sequence_img_id)
                    data_list[sequence_img_id] = {'index': frame_id, 'activity_mat':activity_mat[frame_id - start_f],
                                                    'affordance_mat': affordance_mat[frame_id - start_f],
                                                    'object_mat': object_mat[frame_id - start_f],
                                                    'skeleton_mat': skeleton_mat[frame_id - start_f]}

            with open(os.path.join(corpus_path, category + '.txt'), 'a') as f:
                f.write('* ' +  ' '.join(activity_list) + ' #\n')

    # Align skeleton for each activity
    # Here we only gave code for frame wise data generation
    if option != 'Segment':
        activity_skeleton_map = dict()
        activity_mean_skeleton_map = dict()
        for sequence_id, data_unit in data_list.items():
            activity = metadata.subactivities[int(data_unit['activity_mat'])]
            skeleton = data_unit['skeleton_mat']
            if activity not in activity_skeleton_map.keys():
                activity_skeleton_map[activity] = [skeleton]
            else:
                activity_skeleton_map[activity].append(skeleton)
        for activity, skeletons in activity_skeleton_map.items():
            activity_mean_skeleton_map[activity] = np.mean(np.array(skeletons), axis=0)
        # Align the skeletons according to different activities
        for sequence_id, data_unit in data_list.items():
            activity = metadata.subactivities[int(data_unit['activity_mat'])]
            data_list[sequence_id]['skeleton_mat'] = align_skeleton(data_unit['skeleton_mat'], activity_mean_skeleton_map[activity])

    print('{} out of {} valid videos'.format(valid_count, total_count))

    if option == 'Segment':
        with open(os.path.join(paths.tmp_root, 'video_data_list.p'), 'wb') as f:
            pickle.dump(data_list, f)
    else:
        with open(os.path.join(paths.tmp_root, 'image_data_list.p'), 'wb') as f:
            pickle.dump(data_list, f)

    if option == 'Segment':
        with open(os.path.join(paths.tmp_root, 'video_list.p'), 'wb') as f:
            pickle.dump(available_sequence_ids, f)
    else:
        with open(os.path.join(paths.tmp_root, 'image_list.p'), 'wb') as f:
            pickle.dump(available_sequence_ids, f)

def main():
    paths = vcla_gaze_config.Paths()
    start_time = time.time()
    reformat_data(paths, verbose=False)
    tqdm.write('Time elapsed: {:.2f}s'.format(time.time() - start_time))

if __name__ == '__main__':
    main()