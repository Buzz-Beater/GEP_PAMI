"""
Created on 10/19/18

@author: Baoxiong Jia

Description:

"""
import os
import time
import pickle
import numpy as np
import torch
import torchvision
from skimage import io
import glob
import cv2
import datasets.VCLA_GAZE.vcla_gaze_config as vcla_gaze_config
from datasets.VCLA_GAZE.metadata import VCLA_METADATA
import utils.plyutils as utils
metadata = VCLA_METADATA()

def match_path(img_dir, frame):
    rgb_name = glob.glob(os.path.join(img_dir, 'raw_rgb_{0:05}_*'.format(frame)))
    depth_name = glob.glob(os.path.join(img_dir, 'raw_depth_{0:05}_*'.format(frame)))
    aligned_name = glob.glob(os.path.join(img_dir, 'aligned_rgb_{0:05}.png').format(frame))
    return rgb_name[0], depth_name[0], aligned_name[0]

def get_valid_bbox(bbox):
    x_1 = int(bbox[1])
    y_1 = int(bbox[0])
    x_2 = int(bbox[3])
    y_2 = int(bbox[2])
    return x_1, y_1, x_2, y_2

class VCLA_GAZE_FEATURE(torch.utils.data.Dataset):
    def __init__(self, paths, sequence_ids, transform, input_size, name, task, verbose=False):
        self.root = paths.img_root
        self.tmp_root = paths.tmp_root
        self.inter_root = paths.inter_root
        self.imsize = input_size
        self.name = name
        self.transform = transform
        self.sequence_ids = sequence_ids
        self.task = task
        self.verbose = verbose
        with open(os.path.join(paths.tmp_root, 'image_data_list.p'), 'rb') as f:
            self.data_list = pickle.load(f)

    # Using framewise information for prediction purposes
    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        sequence_info = self.data_list[sequence_id]

        category, video_id, frame = sequence_id.split('$')
        frame = int(frame)

        img_dir = os.path.join(self.root, category, video_id, 'TPV')
        rgb_name, depth_name, aligned_name = match_path(img_dir, frame)

        rgb_image = torch.FloatTensor(io.imread(rgb_name))
        depth_image = torch.FloatTensor(np.array(io.imread(depth_name), dtype=np.double))
        aligned_image = torch.FloatTensor(io.imread(aligned_name))

        activity = torch.LongTensor([sequence_info['activity_mat']])
        object_pair = sequence_info['object_mat']
        object_labels = torch.LongTensor(object_pair[:, :-4])
        bboxs = object_pair[:, -4:]
        object_images = np.empty((1, 3, self.imsize[0], self.imsize[1]))
        for idx, bbox in enumerate(bboxs):
            object_image = np.zeros((3, self.imsize[0], self.imsize[1]), dtype=np.float)
            # Get valid bounding boxes
            x_1, y_1, x_2, y_2 = get_valid_bbox(bbox)
            if np.sum(bbox) != 0:
                bbox_image = rgb_image[y_1 : y_2, x_1 : x_2, :]
                object_image = self.transform(cv2.resize(bbox_image.numpy(), self.imsize, interpolation=cv2.INTER_LINEAR))
            object_images = np.vstack((object_images, np.expand_dims(object_image, axis=0)))
        object_images = torch.FloatTensor(object_images[1:])
        rgb_image = torch.FloatTensor(self.transform(cv2.resize(rgb_image.numpy(), self.imsize, interpolation=cv2.INTER_LINEAR)))
        affordance = torch.LongTensor(sequence_info['affordance_mat'])
        skeleton = torch.FloatTensor(sequence_info['skeleton_mat'])
        if self.task != 'affordance':
            affordance_features = torch.FloatTensor(np.load(os.path.join(self.inter_root, 'finetune', 'affordance', sequence_id + '.npy')))
            assert(affordance_features.shape[0] == 3)
        else:
            affordance_features = torch.Tensor([0])
        if self.verbose:
            return sequence_id, rgb_image, depth_image, aligned_image, activity, object_labels, \
                                        object_images, affordance, skeleton, object_pair
        else:
            return sequence_id, rgb_image, depth_image, aligned_image, activity, object_labels, \
                                        object_images, affordance, skeleton, affordance_features
    def __len__(self):
        return len(self.sequence_ids)


# For testing purposes
def main():
    paths = vcla_gaze_config.Paths()
    start_time = time.time()
    with open(os.path.join(paths.tmp_root, 'image_list.p'), 'rb') as f:
        video_list = pickle.load(f)
    train_ratio = 0.1
    sequence_ids = np.random.permutation(video_list)
    sequence_ids = sequence_ids[:int(train_ratio * len(sequence_ids))]

    input_imsize = (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    training_set = VCLA_GAZE_FEATURE(paths, sequence_ids, transform, input_imsize, 'test', 'activity', verbose=True)

    sequence_id, rgb_image, depth_image, aligned_image, activity, object_labels, \
                                            object_images, affordance, skeleton, object_pair = training_set[0]
    utils.visualize_bbox_rgb(sequence_id, (rgb_image.permute(1, 2, 0), object_pair), metadata.objects)
    utils.visualize_bbox_image(sequence_id, (object_labels, object_images), metadata.objects)
    utils.visualize_skeleton_depth(sequence_id, (aligned_image, skeleton))
    print('Time elapsed: {}s'.format(time.time() - start_time))
    print(sequence_id)


if __name__ == '__main__':
    main()