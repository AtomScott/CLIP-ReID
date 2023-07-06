# encoding: utf-8
"""
@author:  atomscott
@contact: atom.james.scott@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class SynergyReID(BaseImageDataset):
    """
    SynergyReID
    Reference:
    The DeepSportradar Player Re-Identification Challenge (2023)
    URL: https://github.com/DeepSportradar/player-reidentification-challenge/tree/master

    Dataset statistics:
    # identities: 486 (train/test) + 618 (query challenge) + 9505 (gallery challenge)
    # images: 9529 (train/test) + 618 (query challenge) + 9505 (gallery challenge)
    """
    dataset_dir = 'raw/data_reid'

    def __init__(self, root='', verbose=True, pid_begin = 0, use_traintest=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        if use_traintest:
            self.train_dir = None # TODO: Merge train and test sets
            self.query_dir = osp.join(self.dataset_dir, 'reid_challenge/query')
            self.gallery_dir = osp.join(self.dataset_dir, 'reid_challenge/gallery')
            raise NotImplementedError("SynergyReID does not support use_traintest=True")
        else:
            self.train_dir = osp.join(self.dataset_dir, 'reid_training')
            self.query_dir = osp.join(self.dataset_dir, 'reid_test/query')
            self.gallery_dir = osp.join(self.dataset_dir, 'reid_test/gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> SynergyReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        print(self.dataset_dir)
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpeg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)

            # filename should be like 00001_01_01.jpeg so assert it is
            assert filename.count('_') == 2 and filename.endswith('.jpeg'), "Filename is not in the correct format"

            # three numbers indicate pid_trackid_camid.jpeg so split the filename into 3 vars
            basename_without_ext = osp.splitext(filename)[0]  # get filename without extension
            pid, trackid, camid = basename_without_ext.split('_')
            
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            filename = osp.basename(img_path)
            basename_without_ext = osp.splitext(filename)[0]
            pid, trackid, camid = basename_without_ext.split('_')
            
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + int(pid), int(camid), 0))
        return dataset