import os


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.'''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]

def memory(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.'''
    return _VmB('VmStk:') - since


import io
import json
import torch
from math import pi
import numpy as np
from scipy.interpolate import interp1d
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc

from torch.utils.data import Dataset, DataLoader, dataloader

print('import, %.2f MB' % (memory() / 1024 / 1024))

class Comma2k19SequenceDataset(Dataset):
    def __init__(self, split_txt_path, prefix, mode, use_memcache=True):
        self.split_txt_path = split_txt_path
        self.prefix = prefix

        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ('train', 'val')
        self.fix_seq_length = 800 if mode == 'train' else 800

        self.use_memcache = use_memcache
        if self.use_memcache:
            self._init_mc_()

        # from OpenPilot
        self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
        self.t_anchors = np.array(
            (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
             0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
             0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
             2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
             3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
             6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
             8.7890625 ,  9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)

    def __len__(self):
        return len(self.samples)

    def _init_mc_(self):
        from petrel_client.client import Client
        self.client = Client('~/petreloss.conf')
        print('======== Initializing Memcache: Success =======')

    def _get_cv2_vid(self, path):
        print(path)
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
            print(path)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def __getitem__(self, idx):

        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')
        print(seq_sample_path)

        if (cap.isOpened() == False):
            raise RuntimeError
        frame_id = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                success, img_gray_array = cv2.imencode('.png', frame)
                assert(success)
                img_gray_bytes = img_gray_array.tobytes()
                url_to_put = self.prefix + self.samples[idx] + '/frames/%d.png' % frame_id
                if self.client.contains(url_to_put):
                    print('Warning:', url_to_put, 'exists')
                self.client.put(url_to_put, img_gray_bytes)
                frame_id += 1

            else:
                break

        return 1


from tqdm import tqdm

dataset = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 's3://comma2k19/', 'train', use_memcache=True)
tmp_dl = DataLoader(dataset, 12, num_workers=12)
for _ in tqdm(tmp_dl):
    pass

dataset = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 's3://comma2k19/', 'val', use_memcache=True)
tmp_dl = DataLoader(dataset, 12, num_workers=12)
for _ in tqdm(tmp_dl):
    pass
