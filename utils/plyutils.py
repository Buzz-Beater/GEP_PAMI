"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bbox_image(sequence_id, data, objects_dict):
    object_labels, object_images = data
    for obj_idx, obj_label in enumerate(object_labels):
        object_name = objects_dict[np.argmax(obj_label.numpy())]
        fig, ax = plt.subplots(1)
        plt.title(object_name)
        ax.imshow(object_images[obj_idx].permute(1, 2, 0).numpy().astype(np.uint8))
        plt.show()

def visualize_bbox_rgb(sequence_id, data, objects_dict):
    rgb_image, object_pair = data
    color = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(1)
    ax.imshow(rgb_image.numpy().astype(np.uint8))
    plt.title('{} bboxs in rgb'.format(sequence_id))
    for idx, vec in enumerate(object_pair):
        object_id = list(vec[:-4]).index(1)
        if object_id == 0:
            continue
        bbox = vec[-4:]
        # Code for showing wrong bounding boxes
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                 linewidth=1, edgecolor=color[idx], facecolor='none')
        ax.add_patch(rect)
        print(objects_dict[object_id])
    plt.show()


def visualize_skeleton_depth(sequence_id, data):
    image, skeleton = data
    skeleton_depth = skeleton[:, 5 : 7]
    line_pairs = [
                    (23, 11), (24, 11), (11, 10), (10, 9), (9, 8), (8, 20), # right arm
                    (21, 7), (22, 7), (7, 6), (6, 5), (5, 4), (4, 20),      # left arm
                    (3, 2), (2, 20),                                        # head
                    (20, 1), (1, 0),                                        # torso
                    (19, 18), (18, 17), (17, 16), (16, 0),                  # right leg
                    (15, 14), (14, 13), (13, 12), (12, 0)                   # left leg
                  ]
    fig, ax = plt.subplots(1)
    ax.imshow(image.numpy().astype(np.uint8))
    plt.title('{} skeleton in depth'.format(sequence_id))
    for line in line_pairs:
        point1 = [skeleton_depth[line[0], 0], skeleton_depth[line[0], 1]]
        point2 = [skeleton_depth[line[1], 0], skeleton_depth[line[1], 1]]
        ax.scatter(point1[0], point1[1], c='y')
        ax.scatter(point2[0], point2[1], c='y')
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
    plt.show()