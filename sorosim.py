import numpy as np

# file_path = './test.txt'
# rawdata = np.loadtxt(file_path,delimiter=',')
# print(rawdata.shape)
# rawdata[:,3:] = 1000 * rawdata[:,3:]
# np.savez('test2.npz', rawdata)

print(np.load('test2.npz')['arr_0'])