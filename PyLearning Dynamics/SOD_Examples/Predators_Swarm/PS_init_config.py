import numpy as np

def uniform_dist(d, num_points, dist_type, domain):
    if dist_type == 'annulus':
        # Generate random points uniformly in an annulus of given domain
        # 'domain' should be a list or tuple containing the inner and outer radii of the annulus.
        r_inner, r_outer = domain
        r = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.array([x, y])

    elif dist_type == 'disk':
        # Generate random points uniformly in a disk of given radius 'domain'
        r = np.sqrt(np.random.uniform(0, domain**2, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.array([x, y])

    elif dist_type == 'rectangle':
        # Generate random points uniformly in a rectangle of given domain
        # 'domain' should be a list or tuple containing the width and height of the rectangle.
        width, height = domain
        x = np.random.uniform(0, width, num_points)
        y = np.random.uniform(0, height, num_points)
        points = np.array([x, y])

    elif dist_type == 'sphere_surface':
        # Generate random points uniformly on the surface of a sphere with a given radius 'domain'
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = domain * np.sin(phi) * np.cos(theta)
        y = domain * np.sin(phi) * np.sin(theta)
        z = domain * np.cos(phi)
        points = np.array([x, y, z])

    elif dist_type == 'sphere':
        # Generate random points uniformly in a sphere of given radius 'domain'
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = domain * np.cbrt(np.random.uniform(0, 1, num_points))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.array([x, y, z])

    else:
        raise ValueError("Invalid distribution type. Supported types are: 'annulus', 'disk', 'rectangle', 'sphere_surface', 'sphere'")

    return points

def PS_init_config(N, type_info, kind): 
    if kind == 1:
        d = 2
        #print('Uniform distribution parameter \n')
        y_init = np.zeros((d, N))
        type_info = np.array(type_info)

        #preys_ind = type_info == 1
        preys_ind = np.where(type_info ==1,1,0)
        num_preys = np.count_nonzero(preys_ind)
        dist_type = 'annulus'
        l1 = [0.05, 0.15] 
        domain = [i* num_preys for i in l1]
        #print('Uniform distribution parameter \n')
        #print(d, num_preys, dist_type, domain)
        y_init[:, :num_preys] = uniform_dist(d, num_preys, dist_type, domain)

        dist_type = 'disk'
        domain = 0.1
        preds_ind = np.where(type_info ==2,1,0)
        num_preds = np.count_nonzero(preds_ind)
        #print('Uniform distribution parameter \n')
        #print(d, num_preds, dist_type, domain)
        y_init[:, num_preys:] = uniform_dist(d, num_preds, dist_type, domain)

        y_init = y_init.flatten()

    elif kind == 2:
        d = 2
        y_init = np.zeros((d, N))

        epsilon = 0.1
        domain = [epsilon, 1]
        dist_type = 'rectangle'
        preys_ind = np.where(type_info ==1,1,0)
        num_preys = np.count_nonzero(preys_ind)
        y_init[:, preys_ind] = uniform_dist(d, num_preys, dist_type, domain)

        #domain = [0, 0.8 * epsilon]
        l2 = [0, 0.8] 
        domain = [i* epsilon for i in l2]
        preds_ind = np.where(type_info ==2,1,0)
        num_preds = np.count_nonzero(preds_ind)
        y_init[:, preds_ind] = uniform_dist(d, num_preds, dist_type, domain)

        y_init = np.concatenate((y_init.flatten(), np.zeros(d * N)))

    elif kind == 3:
        d = 3
        y_init = np.zeros((d, N))

        preys_ind = np.where(type_info ==1,1,0)
        num_preys = np.count_nonzero(preys_ind)
        dist_type = 'sphere_surface'
        domain = 0.15 * num_preys
        y_init[:, preys_ind] = uniform_dist(d, num_preys, dist_type, domain)

        dist_type = 'sphere'
        domain = 0.1
        preds_ind = np.where(type_info ==2,1,0)
        num_preds = np.count_nonzero(preds_ind)
        y_init[:, preds_ind] = uniform_dist(d, num_preds, dist_type, domain)

        y_init = y_init.flatten()

    return y_init
