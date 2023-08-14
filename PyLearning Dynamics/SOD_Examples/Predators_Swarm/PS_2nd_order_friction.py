import numpy as np

def PS_2nd_order_friction(v, nus, num_classes, class_info):
    d, N = v.shape
    friction = np.zeros((d, N))

    for k in range(1, num_classes + 1):
        agents_Ck1 = class_info == k
        friction[:, agents_Ck1] = -nus[k - 1] * v[:, agents_Ck1]

    friction = friction.flatten()
    return friction
