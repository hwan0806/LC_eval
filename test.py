import numpy as np


# mat = np.array(range(1,10))
# mat2 = np.array(range(11,20))
# print(mat)
# print(mat2)

# mat = mat[:,None]
# mat2 = mat2[:,None]

# mat=(np.concatenate((mat,mat2),axis=1))
# print(mat)
# print(mat[:,1])

# # print(np.stack(mat,mat))

#######################################################################################

# import matplotlib.pyplot as plt

# x = np.array([1,2,3,4,5.5])
# y = np.array([1,2,3,4,5])

# x = x[:,None]
# y = y[:,None]

# mat = np.concatenate((x,y),axis=1)

# t = np.array([0,2,4])
# # print(t.shape)

# # plt.scatter(mat[t,0],mat[t,1])
# # plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x,y)
# ax.scatter(mat[t,0],mat[t,1],c='red')
# plt.show()

#######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# x = np.array([1,2,2,1,1,2,2,1,1])
# y = np.array([1,1,2,2,1,1,2,2,1])
# z = np.array([1,2,3,4,5,6,7,8,9])

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot(x,y,z)

# mat = [[[1,1],[1,1],[1,5]],[[2,1],[2,1],[2,6]]]

# a = 

# # ax.plot([1,1],[1,1],[1,5])
# ax.plot(mat)

# plt.show()

#######################################################################################

a = np.array([[1,1,1],[1,1,1]])
b = np.array([[2,2,2],[2,2,2]])

dist = np.linalg.norm(a - b)

print(dist)