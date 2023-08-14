import numpy as np
import matplotlib.cm as cm

def construct_color_items(K, T, time_vec):
    cmap_names = ['Accent_r', 'Blues','Greens', 'Greens_r', 'Greys', 'Greys_r''Oranges', 'Oranges_r','Purples', 'Purples_r','Reds', 'Reds_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r','viridis', 'viridis_r', 'winter', 'winter_r']
    c_vecs = [np.array([]) for _ in range(K)]
    clabels = []
    cticks = np.zeros(3 * K)
    T_f = time_vec[-1]
    T_0 = time_vec[0]
    c = int(np.ceil(T_f / T))
    t_ticks = [T_0, 4 * T, T_f] if c > 8 else [T_0, T, T_f]
    T0_label = '0' if T_0 == 0 else f'{T_0:.0e}'
    set_labels = [T0_label, 'T', f'{c}T']
    color_shift = T_f * 0.15

    if K > len(cmap_names):
        raise ValueError(f'The Coloring Scheme only works for up to {len(cmap_names)} types of agents!!')

    cmap = np.zeros((K * 256, 4))  # 256 colors in each colormap (default for Matplotlib colormaps)
    for k in range(K):
        ind1 = (k ) * 3 
        ind2 = (k) * 3 + 3
        clabels.extend(set_labels)
        if k == 0:
            cticks[ind1:ind2] = t_ticks
            c_vecs[0] = time_vec
            map1 = cm.get_cmap(cmap_names[0])(np.linspace(0, 1, 256))
            cmap[:256, :] = map1
        else:
            cticks[ind1:ind2] = t_ticks + k * (T_f + color_shift)
            c_vecs[k] = time_vec + k * (T_f + color_shift)
            ind1 = k * 256
            ind2 = k * 256 + 256
            cmap[ind1:ind2, :] = cm.get_cmap(cmap_names[k])(np.linspace(0, 1, 256))

    return {'cmap': cmap, 'c_vecs': c_vecs, 'clabels': clabels, 'cticks': cticks}
