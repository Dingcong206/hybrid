import numpy as np

path = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches_expert/xxx.npy"

x = np.load(path)

print("shape:", x.shape)
print("dtype:", x.dtype)
print("min:", x.min())
print("max:", x.max())
print("mean:", x.mean())
