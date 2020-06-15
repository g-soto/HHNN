import numpy as np

import pickle as pk

with open('resultA.dat','rb') as fa, open('resultB.dat','rb') as fb:
    ra = pk.load(fa)[:,1:].astype(np.float)
    rb = pk.load(fb)[:,1:].astype(np.float)

bestA = ra[:,1:].min(axis=1, keepdims=True)
bestB = rb[:,1:].min(axis=1, keepdims=True)
print('####Test A#####')
print(ra.sum(axis=0))
print(np.sum(bestA))
print(np.mean(bestA - ra, axis=0))

print('\n####Test B#####')
print(rb.sum(axis=0))
print(np.sum(bestB))

print(np.mean(bestB - rb, axis=0))

print(rb[rb[:,0] < np.min(rb[:,1:])])