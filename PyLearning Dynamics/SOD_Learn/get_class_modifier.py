import numpy as np

def get_class_modifier(L, N, d, num_classes, class_info):
    class_mod = np.zeros(N)
    
    for k in range(1, num_classes + 1):
        ind = class_info == k
        class_mod[ind] = np.sqrt(np.count_nonzero(ind))
    
    class_mod = np.repeat(np.kron(class_mod, np.ones(d)), L)
    
    return class_mod
