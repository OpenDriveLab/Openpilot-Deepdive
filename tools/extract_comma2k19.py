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


def debug():
    # we read raw logs via openpilot's tool logreader
    from tools.lib.logreader import LogReader
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Qt5Agg')


    example_segment = 'data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/'

    lr = LogReader(example_segment + 'raw_log.bz2')
    # make list of logs
    logs = list(lr)

    log_types = set([l.which() for l in logs])
    print(log_types)
    # {'pandaStateDEPRECATED', 'androidLog', 'procLog', 'controlsState', 'roadCameraState', 
    #  'roadEncodeIdx', 'longitudinalPlan', 'ubloxGnss', 'radarState', 'carControl', 
    #  'liveLongitudinalMpcDEPRECATED', 'model', 'deviceState', 'liveTracks', 'driverState', 
    #  'sensorEvents', 'sendcan', 'carState', 'gpsLocationExternal', 'clocks', 'ubloxRaw', 
    #  'gpsNMEA', 'qcomGnssDEPRECATD', 'liveMpcDEPRECATED', 'can', 'liveCalibration', 
    #  'gpsLocationDEPRECATED', 'logMessage'}

    # we can plot the speed of the car by getting
    # all the carState logs
    # plt.figsize(12,12)
    # plt.plot([l.carState.vEgo for l in logs if l.which() == 'carState'], linewidth=3)
    # plt.title('Car speed from raw logs (m/s)', fontsize=25)
    # plt.xlabel('boot time (s)', fontsize=18)
    # plt.ylabel('speed (m/s)', fontsize=18)

    # plt.show()

    from tools.lib.framereader import FrameReader
    frame_index = 600

    fr = FrameReader(example_segment + 'video.hevc')
    # plt.imshow(fr.get(frame_index, pix_fmt='rgb24')[0])
    # plt.title('Frame 600 extracted from video with FrameReader', fontsize=25)
    # plt.show()
    frame_count = fr.frame_count

    # 34MB -> 1GB+!! that is too large
    # for i in range(frame_count):
    #     img = fr.get(i, pix_fmt='rgb24')[0]
    #     cv2.imwrite(example_segment + 'ext/%04d.png' % i, img)

    cap = cv2.VideoCapture(example_segment + 'video.hevc')

    if (cap.isOpened() == False):
        print('Error')

    frames = []
    with tqdm() as pbar:
        while (cap.isOpened()):
            pbar.update()
            ret, frame = cap.read()
            if ret == True:
                # cv2.imshow('Frame', frame)
                frames.append(frame)
                
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                    # break

            else:
                break

    cap.release()
    # cv2.destroyAllWindows()


    frame_times = np.load(example_segment + 'global_pose/frame_times')
    print(frame_times.shape)

    print(np.load(example_segment + 'global_pose/frame_orientations').shape)


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


if __name__ == '__main__':
    main()
