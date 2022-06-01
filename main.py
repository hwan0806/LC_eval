import os
import copy
import argparse
from tkinter.messagebox import NO

import numpy as np
from scipy.fft import dst
np.set_printoptions(precision=4)    # 부동소숫점 출력 자리수 결정

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections

#from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# import utils.SCManager as 
from utils.SCManager import *
import utils.UtilsDataset as DtUtils
import utils.UtilsPointcloud as PtUtils
import utils.UtilsVisualization as VsUtils

parser = argparse.ArgumentParser(description='Loop Closing Evaluation')

parser.add_argument('--down_num_points',type=int,default=5000)      # downsample된 point 수 
parser.add_argument('--num_rings',type=int,default=20)              # SC의 행 수 
parser.add_argument('--num_sectors',type=int,default=60)            # SC의 열 수 
parser.add_argument('--num_candidates',type=int,default=1)         # loop detection 후보군 개수 
parser.add_argument('--try_gap_loop_detection',type=int,default=10) # loop detection 과정을 몇 frame 마다 진행할 것인가
parser.add_argument('--loop_threshold',type=float,default=0.11)     # loop detection 결정짓는 경계값
parser.add_argument('--scan_base_dir',type=str,                     # base data directory
                    default='/UbuntuHD1/dataset/KITTI Dataset/Obometry/data_odometry_velodyne/dataset/sequences')
parser.add_argument('--pose_base_dir', type=str,
                    default='/UbuntuHD1/dataset/KITTI Dataset/Obometry/data_odometry_poses/dataset/poses')
parser.add_argument('--sequence_idx',type=str,default='00')         # dataset sequence
parser.add_argument('--LiDAR_name', type=str,default='velodyne')
parser.add_argument('--save_gap',type=int,default=300)              # 
args = parser.parse_args()


# scan dataset 
scan_manager = DtUtils.ScanDirManager(args.scan_base_dir, args.sequence_idx, args.LiDAR_name)
scan_paths = scan_manager.scan_fullpaths       # scan bin file들의 list 
num_frames = len(scan_paths)

SCM = ScanContextManager(shape=[args.num_rings,args.num_sectors],
                         num_candidates=args.num_candidates,
                         threshold=args.loop_threshold,
                         sequence_idx = args.sequence_idx)

pose_manager = DtUtils.PoseDirManager(args.pose_base_dir, args.sequence_idx)
pose = pose_manager.getPose()

src_node = []
dst_node = []
dist_node = []


for for_idx, scan_path in tqdm(enumerate(scan_paths),total=num_frames,mininterval=1.0):
    # point cloud data 처리
    curr_scan_pts = PtUtils.readScan(scan_path)
    curr_scan_down_pts =PtUtils.random_sampling(curr_scan_pts,args.down_num_points)
    
    SCM.addNode(node_idx=for_idx, ptcloud=curr_scan_down_pts)
    
    if(for_idx > 1 and for_idx % args.try_gap_loop_detection == 0):
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if(loop_idx == None): # NOT FOUND
            pass
        else:
            print("Loop event detected: ", for_idx, loop_idx, loop_dist)
            src_node.append(for_idx)
            dst_node.append(loop_idx)
            dist_node.append(loop_dist)

src_node = np.array(src_node)[:,None]
dst_node = np.array(dst_node)[:,None]
dist_node = np.array(dist_node)[:,None]

loop = np.concatenate((src_node,dst_node,dist_node),axis=1)

viz = VsUtils.VizTrajectory(loop,pose)
viz.viz2D()
viz.viz3D()
viz.eval_LC()

plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(pose[:,0],pose[:,1])
# plt.show()


# pose_manager = DtUtils.PoseDirManager(args.pose_base_dir, args.sequence_idx)
# pose = pose_manager.getPose()

# viz = VsUtils.VizTrajectory(0,0,pose)
# viz.viz2D


