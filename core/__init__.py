from .correspond import correspond_mesh
from menpo.shape import PointCloud, TriMesh
import numpy as np
from core.io import import_mesh, getTriMeshfromPly, ply_from_array

# We define a template that we use for the landmarks
LANDMARK_MASK = np.ones(68, dtype=np.bool)
# Remove the jaw...
LANDMARK_MASK[:17] = False
# ...and the inner lips for robustness.
LANDMARK_MASK[-8:] = False

def landmark_and_correspond_mesh(template, mesh, verbose=False, landmark_type='ibug68', LOD = 'low'):
    # Don't touch the original mesh
    mesh = mesh.copy()
    mask = None
    template.landmarks['__lsfm'] = template.landmarks['ibug68'].lms.from_mask(LANDMARK_MASK)
    return_dict = {
        'shape_nicp': correspond_mesh(template, mesh, mask=mask,
                                      verbose=verbose,
                                      landmark_type=landmark_type,
                                      LOD = LOD),
        'mask': None
    }
    return return_dict
