import os
import numpy as np

class ScanDirManager:
    def __init__(self,data_base_dir,sequence_idx,LiDAR_name):
        self.scan_dir = os.path.join(data_base_dir,sequence_idx,LiDAR_name)                         # args 값 받아 directory 이름 완성
        self.scanfile_list = os.listdir(self.scan_dir)                                              # 해당 dir내의 파일명을 list로 만들어 
        self.scanfile_list.sort()                                                                   # 파일명 list를 sort하여 순서대로. 
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scanfile_list]    # bin 파일명까지 붙여 각 frame 파일의 절대경로 완성 
        self.num_scans = len(self.scanfile_list)                                                    # 총 frame 수 계산 
        

class PoseDirManager:
    def __init__(self,pose_base_dir, sequence_idx):
        self.filename = os.path.join(pose_base_dir, sequence_idx) + '.txt'
        self.data = np.loadtxt(self.filename, delimiter=" ",dtype=np.float32)
        self.idx = np.array(range(1,len(self.data[:,0])+1))
    
    def getPose(self):
        # idx = self.idx[:,None]
        x = (self.data[:,3])[:,None]
        y = (self.data[:,11])[:,None]
        z = (0.05 * self.idx)[:,None]
        # z = 7
        pose = np.concatenate((x,y,z),axis=1)
        
        return pose