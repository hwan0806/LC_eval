
from cProfile import label
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VizTrajectory:
    def __init__(self, node, pose):
        self.src = node[:,0].astype(int)
        self.dst = node[:,1].astype(int)
        self.dist = node[:,2]
        self.num_node = len(node)
        
        self.pose = np.array(pose)
        self.x = self.pose[:,0]
        self.y = self.pose[:,1]
        self.z = self.pose[:,2]
        
        self.fig1 = plt.figure(1)
        self.ax1 = self.fig1.add_subplot(111)
        self.fig2 = plt.figure(2)
        self.ax2 = self.fig2.add_subplot(111,projection='3d')
        
        self.TP_src = []
        self.FP_src = []
        self.TP_dst = []
        self.FP_dst = []
        for i,j in zip(self.src,self.dst):
            src_pose = np.array([self.x[i], self.y[i]])
            dst_pose = np.array([self.x[j], self.y[j]])
            if np.linalg.norm(src_pose - dst_pose)<10:
                self.TP_src.append(i)
                self.TP_dst.append(j)
            else:
                self.FP_src.append(i)
                self.FP_dst.append(j)
        
        
        
        
    def viz2D(self):
        x = self.x
        y = self.y
        TruePos = self.TP_src
        FalsePos = self.FP_src
        
    
        self.ax1.plot(x,y)
        
        self.ax1.scatter(self.pose[self.src,0],self.pose[self.src,1],c='red')
        
        # TruePos = []
        # FalsePos = []
        
        # for i,j in zip(self.src,self.dst):
        #     src_pose = np.array([x[i], y[i]])
        #     dst_pose = np.array([x[j], y[j]])
        #     if np.linalg.norm(src_pose - dst_pose)<10:
        #         TruePos.append(i)
        #     else:
        #         FalsePos.append(i)
       
        self.ax1.scatter(self.pose[TruePos,0],self.pose[TruePos,1],c='green')
        self.ax1.scatter(self.pose[FalsePos,0],self.pose[FalsePos,1],c='red')
        
        
        # self.pose[self.src,0] - self.pose[self.dst,0]
        
        
        
        # plt.show()
        
        
    
    def viz3D(self):
        x = self.x
        y = self.y
        z = self.z
        
        self.ax2.plot(x,y,z)
        
        
        for i in range(len(self.TP_src)):
            line_x = [x[self.TP_src[i]],x[self.TP_dst[i]]]
            line_y = [y[self.TP_src[i]],y[self.TP_dst[i]]]
            line_z = [z[self.TP_src[i]],z[self.TP_dst[i]]]
            self.ax2.plot(line_x,line_y,line_z,c='green')
        
        for j in range(len(self.FP_src)):
            line_x = [x[self.FP_src[j]],x[self.FP_dst[j]]]
            line_y = [y[self.FP_src[j]],y[self.FP_dst[j]]]
            line_z = [z[self.FP_src[j]],z[self.FP_dst[j]]]
            self.ax2.plot(line_x,line_y,line_z,c='red')        
        
        # plt.show()
        
    
    def eval_LC(self):
        precision = len(self.TP_dst) / self.num_node * 100
        TP_num = len(self.TP_dst)
        FP_num = len(self.FP_dst)
        
        print(f"Total Loop matching : {self.num_node}, True Positive : {TP_num}, False Positive : {FP_num}, Precision : {precision}%")
        