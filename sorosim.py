import numpy as np

# file_path = './test.txt'
# rawdata = np.loadtxt(file_path,delimiter=',')
# # print(rawdata.shape)
# np.savez('test.npz', rawdata)

print(np.load('data.npz')['arr_0'].shape)