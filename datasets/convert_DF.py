r"""Downloads and converts Market1501 data to TFRecords of TF-Example protos.

This module downloads the Market1501 data, uncompresses it, reads the files
that make up the Market1501 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

try:
    import dataset_utils
except:
    from datasets import dataset_utils
import numpy as np
import pickle
import pdb
import glob
import pandas as pd

# The URL where the Market1501 data can be downloaded.
# _DATA_URL = 'xxxxx'

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0
random.seed(_RANDOM_SEED)

# The number of shards per dataset split.
_NUM_SHARDS = 1

_IMG_PATTERN = '.jpg'


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_folder_path(dataset_dir, split_name):
    if split_name == 'train':
        folder_path = os.path.join(dataset_dir, 'filted_up_train')
    elif split_name == 'train_flip':
        folder_path = os.path.join(dataset_dir, 'filted_up_train_flip')
    elif split_name == 'test':
        folder_path = os.path.join(dataset_dir, 'filted_up_test')
    elif split_name == 'test_samples':
        folder_path = os.path.join(dataset_dir, 'filted_up_test_samples')
    elif split_name == 'test_seq':
        folder_path = os.path.join(dataset_dir, 'filted_up_test_seq')
    assert os.path.isdir(folder_path)
    return folder_path


def _get_image_file_list(dataset_dir, split_name):
    folder_path = _get_folder_path(dataset_dir, split_name)
    if split_name == 'train' or split_name == 'train_flip' or split_name == 'test_samples' or split_name == 'test_seq' or split_name == 'query' or split_name == 'all':
        filelist = sorted(os.listdir(folder_path))
        # filelist = glob.glob(os.path.join(folder_path, _IMG_PATTERN)) # glob will return full path
        # pdb.set_trace()
        filelist = sorted(filelist)
    elif split_name == 'test':
        filelist = (os.listdir(folder_path))
        # filelist = sorted(os.listdir(folder_path))[6617:]  # before 6617 are junk detections
        # filelist = glob.glob(os.path.join(folder_path, _IMG_PATTERN))
        # filelist = sorted(filelist)[6617:]

    # Remove non-jpg files
    valid_filelist = []
    for i in range(0, len(filelist)):
        if filelist[i].endswith('.jpg') or filelist[i].endswith('.png'):
            valid_filelist.append(filelist[i])

    return valid_filelist


def _get_dataset_filename(dataset_dir, out_dir, split_name, shard_id):
    output_filename = 'DeepFashion_%s_%05d-of-%05d.tfrecord' % (
            split_name.split('_')[0], shard_id, _NUM_SHARDS)
    return os.path.join(out_dir, output_filename)


def _get_train_all_pn_pairs(dataset_dir, out_dir, split_name='train', augment_ratio=1, mode='same_diff_cam'):
    """Returns a list of pair image filenames.

    Args:
        dataset_dir: A directory containing person images.

    Returns:
        p_pairs: A list of positive pairs.
        n_pairs: A list of negative pairs.
    """
    assert split_name in {'train', 'train_flip', 'test', 'test_samples', 'test_seq', 'all'}
    if split_name=='train_flip':
        p_pairs_path = os.path.join(out_dir, 'p_pairs_train_flip.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_train_flip.p')
    else:
        p_pairs_path = os.path.join(out_dir, 'p_pairs_'+split_name.split('_')[0]+'.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_'+split_name.split('_')[0]+'.p')
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path,'rb') as f:
            p_pairs = pickle.load(f)
        with open(n_pairs_path,'rb') as f:
            n_pairs = pickle.load(f)
    else:
        p_pairs = []
        n_pairs = []
        pair_lst_path = os.path.join(dataset_dir, "PoseFiltered", "fashion-pairs-test.p")
        with open(pair_lst_path,'rb') as f:
            p_pairs = pickle.load(f)

        # pair_lst_path = os.path.join(dataset_dir,"PoseFiltered","fashion-pairs-test.csv")
        # pair_df = pd.read_csv(pair_lst_path)
        # for index, row in pair_df.iterrows():
        #     source = row['from']
        #     id_s = source[source.find("id") + 2:source.find("id") + 10]
        #     target = row['to']
        #     id_t = target[target.find("id") + 2:target.find("id") + 10]
        #     if id_s == id_t:
        #         p_pairs.append([source, target])
        #     else:
        #         n_pairs.append([source, target])

        # filelist = _get_image_file_list(dataset_dir, split_name)
        # filenames = []
        # pdb.set_trace()
        # if 'test_seq'==split_name:
        #     for i in range(0, len(filelist)):
        #         for j in range(0, len(filelist)):
        #             p_pairs.append([filelist[i],filelist[j]])
        #             if len(p_pairs)%100000==0:
        #                     print(len(p_pairs))
        #
        # elif 'same_diff_cam'==mode:
        #     for i in range(0, len(filelist)):
        #         names = filelist[i].split('_')[0]
        #         id_i = names[names.find("id")+2:names.find("id") + 10]
        #         # names = filelist[i].split('_')
        #         # id_i = names[0]
        #         for j in range(i+1, len(filelist)):
        #             names = filelist[j].split('_')[0]
        #             id_j = names[names.find("id") + 2:names.find("id") + 10]
        #             # names = filelist[j].split('_')
        #             # id_j = names[0]
        #             if id_j == id_i:
        #                 p_pairs.append([filelist[i],filelist[j]])
        #                 p_pairs.append([filelist[j],filelist[i]])  # if two streams share the same weights, no need switch
        #                 if len(p_pairs)%100000==0:
        #                         print(len(p_pairs))
        #             elif j%2000==0 and id_j != id_i:  # limit the neg pairs to 1/40, otherwise it cost too much time
        #                 n_pairs.append([filelist[i],filelist[j]])
        #                 # n_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
        #                 if len(n_pairs)%100000==0:
        #                         print(len(n_pairs))

        print('repeat positive pairs augment_ratio times and cut down negative pairs to balance data ......')
        p_pairs = p_pairs * augment_ratio  
        random.shuffle(n_pairs)
        n_pairs = n_pairs[:len(p_pairs)]
        print('p_pairs length:%d' % len(p_pairs))
        print('n_pairs length:%d' % len(n_pairs))
        print('save p_pairs and n_pairs ......')
        with open(p_pairs_path,'wb') as f:
            pickle.dump(p_pairs,f)
        with open(n_pairs_path,'wb') as f:
            pickle.dump(n_pairs,f)

    print('_get_train_all_pn_pairs finish ......')
    print('p_pairs length:%d' % len(p_pairs))
    print('n_pairs length:%d' % len(n_pairs))

    print('save pn_pairs_num ......')
    pn_pairs_num = len(p_pairs) + len(n_pairs)
    if split_name=='train_flip':
        fpath = os.path.join(out_dir, 'pn_pairs_num_train_flip.p')
    else:
        fpath = os.path.join(out_dir, 'pn_pairs_num_'+split_name.split('_')[0]+'.p')
    with open(fpath,'wb') as f:
        pickle.dump(pn_pairs_num,f)

    return p_pairs, n_pairs


##################### one_pair_rec ###############
import scipy.io
import scipy.stats
import skimage.morphology
from skimage.morphology import square, dilation, erosion
def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    # find connection in the specified sequence, center 29 is in the position 15
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18], [3,17], [6,18]]
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # , [9,12]
    # limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], \
    #            [10,11], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # 
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]] #
    indices = []
    values = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
        
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN>1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


Ratio_0_4 = 1.0/scipy.stats.norm(0, 4).pdf(0)
Gaussian_0_4 = scipy.stats.norm(0, 4)
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)
                elif 'Gaussian'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    if 4==var:
                        values.append( Gaussian_0_4.pdf(distance) * Ratio_0_4  )
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values

def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape

def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        # idx = ind[2]*shape[0]*shape[1] + ind[1]*shape[0] + ind[0]
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape

def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = values[i]
    return dense

def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            # for s in subset:
            #   if s > -1:
            #     cnt += 1
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx>=0:
            peaks = []
            cand_id_list = subsets[valid_idx][:18]
            for ap in all_peaks:
                valid_p = []
                for p in ap:
                    if p[-1] in cand_id_list:
                        valid_p = p
                peaks.append(valid_p)
                # if subsets[valid_idx][i] > -1:
                #   kk = 0
                #   for j in range(valid_idx):
                #     if subsets[j][i] > -1:
                #       kk += 1
                #   peaks.append(all_peaks[i][kk])
                # else:
                #   peaks.append([])

            return all_peaks
        else:
            return None
    except:
        # pdb.set_trace()
        return None

import matplotlib.pyplot as plt 
import scipy.misc
def _visualizePose(pose, img):
    # pdb.set_trace()
    if 3==len(pose.shape):
        pose = pose.max(axis=-1, keepdims=True)
        pose = np.tile(pose, (1,1,3))
    elif 2==len(pose.shape):
        pose = np.expand_dims(pose, -1)
        pose = np.tile(pose, (1,1,3))

    imgShow = ((pose.astype(np.float)+1)/2.0*img.astype(np.float)).astype(np.uint8)
    plt.imshow(imgShow)
    plt.show()


def _format_data(sess, image_reader, folder_path, pairs, i, labels, id_map, attr_mat, id_map_attr, 
                                    all_peaks_dic, subsets_dic, FiltOutMissRegion=False):
    # Read the filename:
    img_path_0 = os.path.join(folder_path, pairs[i][0])
    img_path_1 = os.path.join(folder_path, pairs[i][1])

    # id_0 = pairs[i][0].split('_')[0]
    # id_1 = pairs[i][1].split('_')[0]

    id_0 = img_path_0[img_path_0.find("id") + 2:img_path_0.find("id") + 10]
    id_1 = img_path_1[img_path_1.find("id") + 2:img_path_1.find("id") + 10]
    image_raw_0 = tf.gfile.FastGFile(img_path_0, 'rb').read()
    image_raw_1 = tf.gfile.FastGFile(img_path_1, 'rb').read()
    height, width = image_reader.read_image_dims(sess, image_raw_0)

    attrs_0 = []
    attrs_1 = []
    if attr_mat is not None:
        idx_0 = id_map_attr[id_0]
        idx_1 = id_map_attr[id_1]
        for name in attr_mat.dtype.names:
            attrs_0.append(attr_mat[(name)][0][0][0][idx_0])
            attrs_1.append(attr_mat[(name)][0][0][0][idx_1])

    ########################## Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian))##########################
    ## Pose 16x8
    w_unit = width/16
    h_unit = height/16
    pose_peaks_0 = np.zeros([16,16,18])
    pose_peaks_1 = np.zeros([16,16,18])
    ## Pose coodinate
    pose_peaks_0_rcv = np.zeros([18,3])
    pose_peaks_1_rcv = np.zeros([18,3])
    #
    pose_subs_0 = []
    pose_subs_1 = []
    # pdb.set_trace()
    if (all_peaks_dic is not None) and (pairs[i][0] in all_peaks_dic) and (pairs[i][1] in all_peaks_dic):
        ## Pose 0
        # peaks = all_peaks_dic[pairs[i][0]]
        peaks = _get_valid_peaks(all_peaks_dic[pairs[i][0]], subsets_dic[pairs[i][0]])
        if peaks is None:
            return None
        # print(peaks)
        indices_r4_0, values_r4_0, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_0, shape_0 = _oneDimSparsePose(indices_r4_0, shape)
        indices_r8_0, values_r8_0, shape = _getSparsePose(peaks, height, width, 18, radius=8, mode='Solid')
        indices_r8_0, _ = _oneDimSparsePose(indices_r8_0, shape)
        # pose_dense_r4_0 = _sparse2dense(indices_r4_0, values_r4_0, shape)
        pose_mask_r4_0 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r8_0 = _getPoseMask(peaks, height, width, radius=8, mode='Solid')
        # indices_r6_v4_0, values_r6_v4_0, shape = _getSparsePose(peaks, height, width, 18, radius=6, var=4, mode='Gaussian')
        for ii in range(len(peaks)):
            p = peaks[ii]
            if 0!=len(p):
                pose_peaks_0[int(p[0][1]/h_unit), int(p[0][0]/w_unit), ii] = 1
                pose_peaks_0_rcv[ii][0] = p[0][1]
                pose_peaks_0_rcv[ii][1] = p[0][0]
                pose_peaks_0_rcv[ii][2] = 1
        ## Generate body region proposals
        # part_bbox_list_0, visibility_list_0 = get_part_bbox(peaks, img_path_0, i)
        part_bbox_list_0, visibility_list_0 = get_part_bbox(peaks, img_path_0)
        if FiltOutMissRegion and (0 in visibility_list_0):
            return None
        roi_mask_list_0 = get_roi_mask(part_bbox_list_0, visibility_list_0)
        roi10_mask_0 = np.transpose(np.squeeze(np.array(roi_mask_list_0)),[1,2,0])

        ## Pose 1
        # peaks = all_peaks_dic[pairs[i][1]]
        peaks = _get_valid_peaks(all_peaks_dic[pairs[i][1]], subsets_dic[pairs[i][1]])
        if peaks is None:
            return None
        indices_r4_1, values_r4_1, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)
        indices_r8_1, values_r8_1, shape = _getSparsePose(peaks, height, width, 18, radius=8, mode='Solid')
        indices_r8_1, _ = _oneDimSparsePose(indices_r8_1, shape)
        # pose_dense_r4_1 = _sparse2dense(indices_r4_1, values_r4_1, shape)
        pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r8_1 = _getPoseMask(peaks, height, width, radius=8, mode='Solid')
        # indices_r6_v4_1, values_r6_v4_1, shape = _getSparsePose(peaks, height, width, 18, radius=6, var=4, mode='Gaussian')
        ## Generate body region proposals
        part_bbox_list_1, visibility_list_1 = get_part_bbox(peaks, img_path_1)
        if FiltOutMissRegion and (0 in visibility_list_1):
            return None
        roi_mask_list_1 = get_roi_mask(part_bbox_list_1, visibility_list_1)
        roi10_mask_1 = np.transpose(np.squeeze(np.array(roi_mask_list_1)),[1,2,0])

        ###### Visualize ######
        # dense = _sparse2dense(indices_r4, values_r4, shape)
        # _visualizePose(pose_mask_r4_0, scipy.misc.imread(img_path_0))
        # _visualizePose(pose_mask_r4_1, scipy.misc.imread(img_path_1))
        # if i in [0,5]:
        #     _visualizePose(roi_mask_list_0[0], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[1], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[2], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[3], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[4], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[5], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[6], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[7], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[8], scipy.misc.imread(img_path_0))
        #     _visualizePose(roi_mask_list_0[9], scipy.misc.imread(img_path_0))
        # pdb.set_trace()

        for ii in range(len(peaks)):
            p = peaks[ii]
            if 0!=len(p):
                pose_peaks_1[int(p[0][1]/h_unit), int(p[0][0]/w_unit), ii] = 1
                pose_peaks_1_rcv[ii][0] = p[0][1]
                pose_peaks_1_rcv[ii][1] = p[0][0]
                pose_peaks_1_rcv[ii][2] = 1
        pose_subs_0 = subsets_dic[pairs[i][0]][0].tolist()
        pose_subs_1 = subsets_dic[pairs[i][1]][0].tolist()
    else:
        return None

    example = tf.train.Example(features=tf.train.Features(feature={
            'image_name_0': dataset_utils.bytes_feature(pairs[i][0]),
            'image_name_1': dataset_utils.bytes_feature(pairs[i][1]),
            'image_raw_0': dataset_utils.bytes_feature(image_raw_0),
            'image_raw_1': dataset_utils.bytes_feature(image_raw_1),
            'label': dataset_utils.int64_feature(labels[i]),
            'id_0': dataset_utils.int64_feature(id_map[id_0]),
            'id_1': dataset_utils.int64_feature(id_map[id_1]),
            'cam_0': dataset_utils.int64_feature(-1),
            'cam_1': dataset_utils.int64_feature(-1),
            'image_format': dataset_utils.bytes_feature('jpg'),
            'image_height': dataset_utils.int64_feature(height),
            'image_width': dataset_utils.int64_feature(width),
            'real_data': dataset_utils.int64_feature(1),
            'attrs_0': dataset_utils.int64_feature(attrs_0),
            'attrs_1': dataset_utils.int64_feature(attrs_1),
            'pose_peaks_0': dataset_utils.float_feature(pose_peaks_0.flatten().tolist()),
            'pose_peaks_1': dataset_utils.float_feature(pose_peaks_1.flatten().tolist()),
            'pose_peaks_0_rcv': dataset_utils.float_feature(pose_peaks_0_rcv.flatten().tolist()),
            'pose_peaks_1_rcv': dataset_utils.float_feature(pose_peaks_1_rcv.flatten().tolist()),
            # 'pose_dense_r4_0': dataset_utils.int64_feature(pose_dense_r4_0.astype(np.int64).flatten().tolist()),
            # 'pose_dense_r4_1': dataset_utils.int64_feature(pose_dense_r4_1.astype(np.int64).flatten().tolist()),
            'pose_mask_r4_0': dataset_utils.int64_feature(pose_mask_r4_0.astype(np.int64).flatten().tolist()),
            'pose_mask_r4_1': dataset_utils.int64_feature(pose_mask_r4_1.astype(np.int64).flatten().tolist()),
            'pose_mask_r8_0': dataset_utils.int64_feature(pose_mask_r8_0.astype(np.int64).flatten().tolist()),
            'pose_mask_r8_1': dataset_utils.int64_feature(pose_mask_r8_1.astype(np.int64).flatten().tolist()),

            'shape': dataset_utils.int64_feature(shape_0),

            # 'indices_r6_v4_0': dataset_utils.int64_feature(np.array(indices_r6_v4_0).astype(np.int64).flatten().tolist()),
            # 'values_r6_v4_0': dataset_utils.float_feature(np.array(values_r6_v4_0).astype(np.float).flatten().tolist()),
            # 'indices_r6_v4_1': dataset_utils.int64_feature(np.array(indices_r6_v4_1).astype(np.int64).flatten().tolist()),
            # 'values_r6_v4_1': dataset_utils.float_feature(np.array(values_r6_v4_1).astype(np.float).flatten().tolist()),
            
            'indices_r4_0': dataset_utils.int64_feature(np.array(indices_r4_0).astype(np.int64).flatten().tolist()),
            'values_r4_0': dataset_utils.float_feature(np.array(values_r4_0).astype(np.float).flatten().tolist()),
            'indices_r4_1': dataset_utils.int64_feature(np.array(indices_r4_1).astype(np.int64).flatten().tolist()),
            'values_r4_1': dataset_utils.float_feature(np.array(values_r4_1).astype(np.float).flatten().tolist()),
            'indices_r8_0': dataset_utils.int64_feature(np.array(indices_r8_0).astype(np.int64).flatten().tolist()),
            'values_r8_0': dataset_utils.float_feature(np.array(values_r8_0).astype(np.float).flatten().tolist()),
            'indices_r8_1': dataset_utils.int64_feature(np.array(indices_r8_1).astype(np.int64).flatten().tolist()),
            'values_r8_1': dataset_utils.float_feature(np.array(values_r8_1).astype(np.float).flatten().tolist()),

            'pose_subs_0': dataset_utils.float_feature(pose_subs_0),
            'pose_subs_1': dataset_utils.float_feature(pose_subs_1),

            'part_bbox_0': dataset_utils.int64_feature(np.array(part_bbox_list_0).astype(np.int64).flatten().tolist()),
            'part_bbox_1': dataset_utils.int64_feature(np.array(part_bbox_list_1).astype(np.int64).flatten().tolist()),
            'part_vis_0': dataset_utils.int64_feature(np.array(visibility_list_0).astype(np.int64).flatten().tolist()),
            'part_vis_1': dataset_utils.int64_feature(np.array(visibility_list_1).astype(np.int64).flatten().tolist()),
            'roi10_mask_0': dataset_utils.int64_feature(roi10_mask_0.astype(np.int64).flatten().tolist()),
            'roi10_mask_1': dataset_utils.int64_feature(roi10_mask_1.astype(np.int64).flatten().tolist()),
    }))

    return example

def get_part_bbox(peaks, img_path=None, idx=None, img_H=256, img_W=256):
    ## Generate body region proposals
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear [0,1,2,5,14,15,16,17]
    ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip [2,3,4,5,6,7,8,11]
    ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank [8,9,10,11,12,13]
    ## part4: Lsho, Lelb, Lwri [3,6,7]
    ## part5: Rsho, Relb, Rwri [2,4,5]
    ## part6: Lhip, Lkne, Lank [11,12,13]
    ## part7: Rhip, Rkne, Rank [8,9,10]
    ###################################
    ## part8: Rsho, Lsho, Rhip, Lhip [2,5,8,11]
    ## part9: Lsho, Lelb [5,6]
    ## part10: Lelb, Lwri [6,7]
    ## part11: Rsho, Relb [2,3]
    ## part12: Relb, Rwri [3,4]
    ## part13: Lhip, Lkne [11,12]
    ## part14: Lkne, Lank [12,13]
    ## part15: Rhip, Rkne [8,9]
    ## part16: Rkne, Rank [9,10]
    ## part17: WholeBody range(0,18)
    ## part18-36: single key point [0],...,[17]
    ## part36: Rsho, Relb, Rwri, Rhip, Rkne, Rank [2,3,4,8,9,10]
    ## part37: Lsho, Lelb, Lwri, Lhip, Lkne, Lank [5,6,7,11,12,13]
    part_idx_list_all = [ [0,1,2,5,14,15,16,17], ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear
                        [2,3,4,5,6,7,8,11], ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip
                        [8,9,10,11,12,13], ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank
                        [5,6,7], ## part4: Lsho, Lelb, Lwri
                        [2,3,4], ## part5: Rsho, Relb, Rwri
                        [11,12,13], ## part6: Lhip, Lkne, Lank
                        [8,9,10], ## part7: Rhip, Rkne, Rank
                        [2,5,8,11], ## part8: Rsho, Lsho, Rhip, Lhip
                        [5,6], ## part9: Lsho, Lelb
                        [6,7], ## part10: Lelb, Lwri
                        [2,3], ## part11: Rsho, Relb
                        [3,4], ## part12: Relb, Rwri
                        [11,12], ## part13: Lhip, Lkne
                        [12,13], ## part14: Lkne, Lank
                        [8,9], ## part15: Rhip, Rkne
                        [9,10], ## part16: Rkne, Rank
                        range(0,18) ] ## part17: WholeBody
    part_idx_list_all.extend([[i] for i in range(0,18)]) ## part18-35: single key point
    part_idx_list_all.extend([ [2,3,4,8,9,10], ## part36: Rsho, Relb, Rwri, Rhip, Rkne, Rank
                        [5,6,7,11,12,13]]) ## part37: Lsho, Lelb, Lwri, Lhip, Lkne, Lank
    # part_idx_list = [part_idx_list_all[i] for i in [0,1,2,3,4,5,6,7,8,16]] ## select >3 keypoints
    part_idx_list = part_idx_list_all ## select all
    part_bbox_list = [] ## bbox: normalized coordinates [y1, x1, y2, x2]
    visibility_list = []
    ## Judge wheather it's whole body or not
    for ii in range(len(part_idx_list)):
        part_idx = part_idx_list[ii]
        xs = []
        ys = []
        select_peaks = [peaks[i] for i in part_idx]
        for p in select_peaks:
            if 0!=len(p):
                xs.append(p[0][0])
                ys.append(p[0][1])
        if len(xs)==0:
            visibility_list.append(0)
        else:
            visibility_list.append(1)

    if visibility_list[13] and visibility_list[15]:
        ## Whole body that includes the following two parts
        ## part14: Lkne, Lank [12,13]
        ## part16: Rkne, Rank [9,10]
        WholeBody = True
        r = 10
        r_single = 20
    else:
        WholeBody = False
        r = 20
        r_single = 40

    for ii in range(len(part_idx_list)):
        part_idx = part_idx_list[ii]
        xs = []
        ys = []
        select_peaks = [peaks[i] for i in part_idx]
        for part_id in part_idx:
            p = peaks[part_id]
            if 0!=len(p):
                x = p[0][0]
                y = p[0][1]

                if part_id in [2,5]:
                    # if 0==ii: ## head+shoulder
                    pass
                if part_id in [0]: ## enlarge the head roi mask
                    if WholeBody:
                        y  = max(0,y-10)
                    else:
                        y  = max(0,y-25)
                elif part_id in [3,4,6,35,36]: ## enlarge the wrist and ankle roi mask
                    # if WholeBody:
                    #     y  = min(img_H-1,y+5)
                    # else:
                    #     y  = min(img_H-1,y+10)
                    pass

                # if not WholeBody:
                #     y1_t  = max(0,y1_t-5)
                #     x1_t  = max(0,x1_t-5)
                #     y2_t  = min(img_H-1,y2_t+5)
                #     x2_t  = min(img_W-1,x2_t+5)

                xs.append(x)
                ys.append(y)
        if len(xs)==0:
            part_bbox_list.append([0,0,1,1])
        else:
            y1 = np.array(ys).min()
            x1 = np.array(xs).min()
            y2 = np.array(ys).max()
            x2 = np.array(xs).max()
            if len(xs)>1:
                y1 = max(0,y1-r)
                x1 = max(0,x1-r)
                y2 = min(img_H-1,y2+r)
                x2 = min(img_W-1,x2+r)
            else:
                y1 = max(0,y1-r_single)
                x1 = max(0,x1-r_single)
                y2 = min(img_H-1,y2+r_single)
                x2 = min(img_W-1,x2+r_single)
            part_bbox_list.append([y1, x1, y2, x2])
        if idx is not None:
            img = scipy.misc.imread(img_path)
            scipy.misc.imsave('%04d_part%d.jpg'%(idx,ii+1), img[y1:y2,x1:x2,:])

    if idx is not None:
        scipy.misc.imsave('%04d_part_whole.jpg'%idx, img)

    return part_bbox_list, visibility_list

def get_roi_mask(part_bbox_list, visibility_list, img_H=256, img_W=256):
    ## Generate body region proposals
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    ######### small region ############
    ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear [0,1,2,5,14,15,16,17]
    ## part4: Lsho, Lelb, Lwri [3,6,7]
    ## part5: Rsho, Relb, Rwri [2,4,5]
    ## part6: Lhip, Lkne, Lank [11,12,13]
    ## part7: Rhip, Rkne, Rank [8,9,10]
    ########## big region ############
    ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip [2,3,4,5,6,7,8,11]
    ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank [8,9,10,11,12,13]
    ## part36: Rsho, Relb, Rwri, Rhip, Rkne, Rank [2,3,4,8,9,10]
    ## part37: Lsho, Lelb, Lwri, Lhip, Lkne, Lank [5,6,7,11,12,13]
    ## part1+2: nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip, Leye, Reye, Lear, Rear [0,1,2,3,4,5,6,7,8,11,14,15,16,17]
    roi_num = 5
    if visibility_list[13] and visibility_list[15]:
        ## Whole body that includes the following two parts
        ## part14: Lkne, Lank [12,13]
        ## part16: Rkne, Rank [9,10]
        WholeBody = True
    else:
        WholeBody = False

    if WholeBody:
        bbox_small_idxs_list = [[0],[3],[4],[5],[6]]
        bbox_big_idxs_list = [[1],[2],[35],[36],[0,1]]
    else:
        bbox_small_idxs_list = [[0],[3],[4],[3],[4]]
        bbox_big_idxs_list = [[1],[35],[36],[35],[36]]

    roi_small_mask_list = []
    for bbox_idxs in bbox_small_idxs_list:
        y1, x1, y2, x2 = [img_H-1, img_W-1, 0, 0]
        valid = False
        for part_idx in bbox_idxs:
            if visibility_list[part_idx]:
                valid = True
                y1_t, x1_t, y2_t, x2_t = part_bbox_list[part_idx]
                if part_idx==0: ## enlarge the head roi mask
                    if WholeBody:
                        y1_t  = max(0,y1_t-10)
                    else:
                        y1_t  = max(0,y1_t-20)
                elif part_idx in [3,4,5,6,2,35,36]: ## enlarge the wrist and ankle roi mask
                    y2_t  = min(img_H-1,y2_t+20)

                if not WholeBody:
                    y1_t  = max(0,y1_t-5)
                    x1_t  = max(0,x1_t-5)
                    y2_t  = min(img_H-1,y2_t+5)
                    x2_t  = min(img_W-1,x2_t+5)

                y1 = min(y1, y1_t)
                x1 = min(x1, x1_t)
                y2 = max(y2, y2_t)
                x2 = max(x2, x2_t)
        if valid:
            mask = np.ones([img_H, img_W, 1])
            mask[y1:y2,x1:x2] *= 0
            roi_small_mask_list.append(mask)

    while len(roi_small_mask_list)<roi_num:
        rand_idx = int(np.random.choice(len(roi_small_mask_list),1)-1)
        roi_small_mask_list.append(roi_small_mask_list[rand_idx])

    roi_big_mask_list = []
    for bbox_idxs in bbox_big_idxs_list:
        y1, x1, y2, x2 = [img_H-1, img_W-1, 0, 0]
        valid = False
        for part_idx in bbox_idxs:
            if visibility_list[part_idx]:
                valid = True
                y1_t, x1_t, y2_t, x2_t = part_bbox_list[part_idx]
                if part_idx==0: ## enlarge the head roi mask
                    if WholeBody:
                        y1_t  = max(0,y1_t-10)
                    else:
                        y1_t  = max(0,y1_t-20)
                elif part_idx in [3,4,5,6,2,35,36]: ## enlarge the wrist and ankle roi mask
                    y2_t  = min(img_H-1,y2_t+20)

                if not WholeBody:
                    y1_t  = max(0,y1_t-5)
                    x1_t  = max(0,x1_t-5)
                    y2_t  = min(img_H-1,y2_t+5)
                    x2_t  = min(img_W-1,x2_t+5)

                y1 = min(y1, y1_t)
                x1 = min(x1, x1_t)
                y2 = max(y2, y2_t)
                x2 = max(x2, x2_t)
        if valid:
            mask = np.ones([img_H, img_W, 1])
            mask[y1:y2,x1:x2] *= 0
            roi_big_mask_list.append(mask)

    while len(roi_big_mask_list)<roi_num:
        rand_idx = int(np.random.choice(len(roi_big_mask_list),1)-1)
        roi_big_mask_list.append(roi_big_mask_list[rand_idx])

    roi_mask_list = roi_small_mask_list + roi_big_mask_list

    assert((2*roi_num)==len(roi_mask_list))
    return roi_mask_list


def _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, 
            pose_peak_path=None, pose_sub_path=None, pose_peak_path_flip=None, pose_sub_path_flip=None, tf_record_pair_num=np.inf):
    """Converts the given pairs to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        pairs: A list of image name pairs.
        labels: label list to indicate positive(1) or negative(0)
        dataset_dir: The directory where the converted datasets are stored.
    """
    if split_name_flip is None:
        USE_FLIP = False
    else:
        USE_FLIP = True

    # num_shards = _NUM_SHARDS
    num_shards = 1
    assert split_name in ['train', 'test', 'test_samples', 'test_seq']
    num_per_shard = int(math.ceil(len(pairs) / float(num_shards)))
    folder_path = _get_folder_path(dataset_dir, split_name)
    if USE_FLIP:
        folder_path_flip = _get_folder_path(dataset_dir, split_name_flip)

    # Load attr mat file
    attr_mat = None
    id_map_attr = None

    # Load pose pickle file 
    all_peaks_dic = None
    subsets_dic = None
    all_peaks_dic_flip = None
    subsets_dic_flip = None
    with open(pose_peak_path, 'rb') as f:
        all_peaks_dic = pickle.load(f,encoding='bytes')
    with open(pose_sub_path, 'rb') as f:
        subsets_dic = pickle.load(f,encoding='bytes')
    if USE_FLIP:
        with open(pose_peak_path_flip, 'rb') as f:
            all_peaks_dic_flip = pickle.load(f,encoding='bytes')
        with open(pose_sub_path_flip, 'rb') as f:
            subsets_dic_flip = pickle.load(f,encoding='bytes')
    # Transform ids to [0, ..., num_of_ids]
    id_cnt = 0
    id_map = {}
    for i in range(0, len(pairs)):
        id_0 = pairs[i][0][pairs[i][0].find("id") + 2:pairs[i][0].find("id") + 10]
        id_1 = pairs[i][1][pairs[i][1].find("id") + 2:pairs[i][1].find("id") + 10]
        # id_0 = pairs[i][0].split('_')[0]
        # id_1 = pairs[i][1].split('_')[0]
        if id_0 not in id_map.keys():
        # if not id_map.has_key(id_0):
            id_map[id_0] = id_cnt
            id_cnt += 1
        if id_1 not in id_map.keys():
            id_map[id_1] = id_cnt
            id_cnt += 1
    print('id_map length:%d' % len(id_map))
    if USE_FLIP:
        id_cnt = 0
        id_map_flip = {}
        for i in range(0, len(pairs_flip)):
            id_0 = pairs_flip[i][0].split('_')[0]
            id_1 = pairs_flip[i][1].split('_')[0]
            if not id_map_flip.has_key(id_0):
                id_map_flip[id_0] = id_cnt
                id_cnt += 1
            if not id_map_flip.has_key(id_1):
                id_map_flip[id_1] = id_cnt
                id_cnt += 1
        print('id_map_flip length:%d' % len(id_map_flip))
    # Pairs = pairs
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                        dataset_dir, out_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    cnt = 0

                    if USE_FLIP:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id+1) * num_per_shard, len(pairs_flip))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                    i+1, len(pairs_flip), shard_id))
                            sys.stdout.flush()
                            example = _format_data(sess, image_reader, folder_path_flip, pairs_flip, i, labels_flip, id_map_flip, attr_mat, id_map_attr, 
                                                                        all_peaks_dic_flip, subsets_dic_flip)
                            if None==example:
                                continue
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1
                            if cnt==tf_record_pair_num:
                                break

                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(pairs))
                    for i in range(start_ndx, end_ndx):
                        if i > len(pairs) - 1:
                            break
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(pairs), shard_id))
                        sys.stdout.flush()
                        example = _format_data(sess, image_reader, folder_path, pairs, i, labels, id_map, attr_mat, id_map_attr, all_peaks_dic, subsets_dic)
                        if None==example:
                            print(pairs[i])
                            pairs.pop(i)
                            continue
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1
                        if cnt==tf_record_pair_num:
                            break

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('cnt:',cnt)
    with open(os.path.join(out_dir,'tf_record_pair_num.txt'),'w') as f:
        f.write('cnt:%d' % cnt)
    # p_pairs_path = os.path.join(out_dir, 'p_pairs_' + split_name.split('_')[0] + '.p')
    # with open(p_pairs_path,'wb') as f:
    #     pickle.dump(pairs,f)
    #
    # fpath = os.path.join(out_dir, 'pn_pairs_num_'+split_name.split('_')[0]+'.p')
    # with open(fpath, 'wb') as f:
    #     pickle.dump(len(pairs), f)


def run_one_pair_rec(dataset_dir, out_dir, split_name):
    # if not tf.gfile.Exists(dataset_dir):
    #     tf.gfile.MakeDirs(dataset_dir)
    
    if split_name.lower()=='train':
        #================ Prepare training set ================
        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        pose_peak_path_flip = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion_Flip.p')
        pose_sub_path_flip = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion_Flip.p')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name=split_name,
                                                    augment_ratio=1, 
                                                    mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        split_name_flip='train_flip'
        p_pairs_flip, n_pairs_flip = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name=split_name_flip,
                                                    augment_ratio=1, 
                                                    mode='same_diff_cam')
        p_labels_flip = [1]*len(p_pairs_flip)
        n_labels_flip = [0]*len(n_pairs_flip)
        pairs_flip = p_pairs_flip
        labels_flip = p_labels_flip
        combined = list(zip(pairs_flip, labels_flip))
        random.shuffle(combined)
        pairs_flip[:], labels_flip[:] = zip(*combined)


        # print('os.remove pn_pairs_num_train_flip.p')
        # os.remove(os.path.join(out_dir, 'pn_pairs_num_train_flip.p'))

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, 
          pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path, pose_peak_path_flip=pose_peak_path_flip, pose_sub_path_flip=pose_sub_path_flip)

        print('\nTrain convert Finished !')

    if split_name.lower()=='test_samples':
        # ================ Prepare test_samples set ================
        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                  split_name=split_name,
                                                  augment_ratio=1, 
                                                  mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name,split_name_flip, pairs, pairs_flip, labels, labels_flip, 
            dataset_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path)

        print('\nTest samples convert Finished !')

    if split_name.lower()=='test':
        # ================ Prepare test set ================
        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                  split_name=split_name,
                                                  augment_ratio=1, 
                                                  mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        # random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name,split_name_flip, pairs, pairs_flip, labels, labels_flip, 
            dataset_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path)

        print('\nTest samples convert Finished !')

    # ================ Prepare test_seq set ================
    if split_name.lower()=='test_seq':
        pose_peak_path = os.path.join(dataset_dir,'PoseFiltered','all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir,'PoseFiltered','subsets_dic_DeepFashion.p')
        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                split_name=split_name,
                                                augment_ratio=1, 
                                                mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None

        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name,split_name_flip, pairs, pairs_flip, labels, labels_flip, 
            dataset_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path)

        print('\nTest seq convert Finished !')


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    split_name = sys.argv[2]   ## 'train', 'test', 'test_samples', 'test_seq'
    out_dir = os.path.join(dataset_dir, 'DF_'+split_name.replace('_flip','')+'_data')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run_one_pair_rec(dataset_dir, out_dir, split_name)


# CUDA_VISIBLE_DEVICES=7 python convert_market.py './data/Market1501_img_pose_attr_seg' 'test'