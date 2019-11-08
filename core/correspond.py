#from functools import lru_cache
import numpy as np
from core.nicp import non_rigid_icp
from menpo.shape import PointCloud, TriMesh
#from .data import load_template
import csv
import os.path
from core.configuration import paramList

def smootherstep(x, x_min, x_max):
    y = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * (y ** 5) - 15 * (y ** 4) + 10 * (y ** 3)


def generate_data_weights(template, nosetip, r_mid=1.05, r_width=0.3,
                          y_pen=1.4, w_inner=1, w_outer=0):
    r_min = r_mid - (r_width / 2)
    r_max = r_mid + (r_width / 2)
    w_range = w_inner - w_outer
    x = np.sqrt(np.sum((template.points - nosetip.lms.points) ** 2 *
                       np.array([1, y_pen, 1]), axis=1))
    return ((1 - smootherstep(x, r_min, r_max))[:, None] * w_range + w_outer).T


def generate_data_weights_per_iter(template, nosetip, r_width, w_min_iter,
                                   w_max_iter, r_mid=10.5, y_pen=1.4, LOD = 'low'):
    # Change in the data term follows the same pattern that is used for the
    # stiffness weights
    param = paramList(LOD = LOD)
    stiffness_weights = param.stiffness_weights
    #stiffness_weights = np.array([50, 20, 5, 2, 0.8])
    #stiffness_weights = np.array([50, 40, 30, 20, 5, 2, 0.8, 0.5, 0.35, 0.2, 0.1, 0.05])
    #stiffness_weights = np.array([ 10, 5 ])
    s_iter_range = stiffness_weights[0] - stiffness_weights[-1]
    w_iter_range = w_max_iter - w_min_iter
    m = w_iter_range / s_iter_range
    c = w_max_iter - m * stiffness_weights[0]
    w_outer = m * stiffness_weights + c
    w_inner = 1
    return generate_data_weights(template, nosetip, w_inner=w_inner,
                                 w_outer=w_outer, r_width=r_width, r_mid=r_mid,
                                 y_pen=y_pen)


#@lru_cache()
def data_weights(template, LOD = 'low'):
    w_max_iter = 0.5
    w_min_iter = 0.0
    r_width = 0.5 * 0.84716526594210229
    r_mid = 0.95 * 0.84716526594210229
    y_pen = 1.7
    return generate_data_weights_per_iter(template,
                                          template.landmarks['nosetip'],
                                          r_width=r_width,
                                          r_mid=r_mid,
                                          w_min_iter=w_min_iter,
                                          w_max_iter=w_max_iter,
                                          y_pen=y_pen,
                                          LOD=LOD,
                                          )


def correspond_mesh(template, mesh, mask=None, verbose=False, landmark_type = 'ibug68', LOD = 'low'):
    group = 'unre_pickpoints'
    template.landmarks['unre_pickpoints'] = template.landmarks[landmark_type]
    param = paramList(LOD = LOD)
    stiffness_weights = param.stiffness_weights
    landmark_weights = param.landmark_weights
    aligned = non_rigid_icp(template, mesh, landmark_group=group, eps=5e-3,
                            stiffness_weights = stiffness_weights, landmark_weights = landmark_weights,
                            data_weights=data_weights(template, LOD = LOD), verbose=verbose)
    return aligned
