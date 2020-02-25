import numpy as np


file = np.genfromtxt('data.txt', delimiter=',')
file = file.astype('int32')
# print(file > 5)         # Boolean Masking
print(file)
print(file [~(file > 5)])
print((file % 2))
x = np.all(file > 5, axis=1)
print(x)

# a = np.random.randint(100, size=(2,2))
# b = np.random.randint(200, size=(2,2))
# print(a)
# print(b)
# mx_a = np.max(a, axis=1)        # Max Min Sum
# mx_b = np.min(b, axis=1)
# c = np.array([mx_a, mx_b])
# print(c)
# print(np.sum(c, axis=1))
# new_b = b.reshape((1,4))
# print(new_b)
# a = np.array([1,2,3])
# aa_stack = np.hstack([a,a])     # v & h stack
# print(aa_stack)

# a = np.full((2,3), 7)
# b = np.full((3,2), 8)
# print(np.matmul(a,b))   # Matrix Multiply
# c = np.identity(3)
# print(np.linalg.det(c))


# a = np.array([1,2,3,4], dtype='int64')
# print(a**20)
# print(np.sin(a))

# m = np.ones((7,7))
# n = np.zeros((3,3))
# n[1,1] = 9
# m[2:5, 2:5] = n
# print(m)
# mm = m.copy()
# print(mm)
# a = np.array([[1,2,3,4]])
# r = np.repeat(a, 3, axis=0)     # 0 = vertical, 1 = horizontal
# print(r)

# x = np.full((2,2,2), (2,3))
# x[:,1,:]=[7,8]
# print(x)
# b = np.full_like(x, 4)    # = np.full(x.shape, 4)
# rn = np.random.randint(-8,8, size=(3,3))
# print(rn)

# a = np.array([[1,2,3,4],[5,6,7,8]], dtype='int16')
# a[:, 2] = a[:, 3]
# print(a[1, 0:4:2])    # Access with interval
# print(a[:, 2:4])      # Accessing elements
# print(a.ndim)       # Dimension
# print(a.shape)      # Shape of array
# print(a.dtype)      # Data type
# print(a.itemsize)   # Size of per item
# print(a.size)       # Total Items

b = np.array([[8,7,6,5], [4,3,2,1]])
# x = np.vstack([a,b,b,a])
# print(x)