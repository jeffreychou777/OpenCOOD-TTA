import numpy as np

# 创建一个形状为 (10000, 4) 的数组
array = np.arange(10000 * 4).reshape(10000, 4)

# 定义需要分割的长度列表
split_lengths = [1000, 3000, 2000, 4000]

# 使用 np.split 结合 cumsum 来按给定的长度分割数组
split_array_list = np.split(array, np.cumsum(split_lengths)[:-1])

# 检查每个数组的形状
shapes = [arr.shape for arr in split_array_list]
print(shapes)
