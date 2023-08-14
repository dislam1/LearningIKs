import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
def get_colored_line(c1_at, c2_at, c, ax):
    #N = x.shape[0]
    N=500
    for i in range(len(c1_at)):
        x = c1_at[i]
        y = c2_at[i]

        xx = np.linspace(x.min(), x.max(), N)
        yy = np.interp(xx, x, y)
        cc = np.interp(xx, x, c)
        
        ax = plt.gca()
        
        points = np.array([xx, yy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(yy.min(), yy.max())
        #lc = LineCollection(segments, cmap='viridis', norm=norm)
        #lc.set_array(cc)
        #line = ax.add_collection(lc)
        
        dots = ax.scatter(x, y, c=y)
    
  