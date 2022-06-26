from __future__ import print_function

from tqdm import tqdm
import os
import cv2
import glob
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import random
random.seed(0)


def main():
    sequences = glob.glob('data/comma2k19/*/*/*/video.hevc')
    random.shuffle(sequences)
    
    num_seqs = len(sequences)
    print(num_seqs, 'sequences')

    num_train = int(0.8 * num_seqs)

    with open('data/comma2k19_train.txt', 'w') as f:
        f.writelines(seq.replace('data/comma2k19/', '').replace('/video.hevc', '\n') for seq in sequences[:num_train])
    with open('data/comma2k19_val.txt', 'w') as f:
        f.writelines(seq.replace('data/comma2k19/', '').replace('/video.hevc', '\n') for seq in sequences[num_train:])
    example_segment = 'data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/'
    frame_times = np.load(example_segment + 'global_pose/frame_times')
    print(frame_times.shape)

    # === Generating non-overlaping seqs ===
    sequences = glob.glob('data/comma2k19/*/*/*/video.hevc')
    sequences = [seq.replace('data/comma2k19/', '').replace('/video.hevc', '') for seq in sequences]
    seq_names = list(set([seq.split('/')[1] for seq in sequences]))
    num_seqs = len(seq_names)
    num_train = int(0.8 * num_seqs)
    train_seq_names = seq_names[:num_train]
    with open('data/comma2k19_train_non_overlap.txt', 'w') as f:
        f.writelines(seq + '\n' for seq in sequences if seq.split('/')[1] in train_seq_names)
    with open('data/comma2k19_val_non_overlap.txt', 'w') as f:
        f.writelines(seq + '\n' for seq in sequences if seq.split('/')[1] not in train_seq_names)


if __name__ == '__main__':
    main()
