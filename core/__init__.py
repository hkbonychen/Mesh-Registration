from .landmark import landmark_mesh
from .correspond import correspond_mesh
import os_io as lio

def landmark_and_correspond_mesh(template, mesh, verbose=False, landmark_type='ibug68'):
    # Don't touch the original mesh
    mesh = mesh.copy()

    if hasattr(mesh, 'texture'):
        lms = landmark_mesh(template, mesh, verbose=True)
        mesh.landmarks['__lsfm_masked'] = lms['landmarks_3d_masked']
        mask = lms['occlusion_mask']
        return_dict = {
            'shape_nicp': correspond_mesh(template, mesh, mask=mask,
                                          verbose=verbose,
                                          landmark_type=landmark_type),
            'landmarked_image': lms['landmarked_image'],
            'mask': mask
        }
    else:
        mask = None
        return_dict = {
            'shape_nicp': correspond_mesh(template, mesh, mask=mask,
                                          verbose=verbose,
                                          landmark_type=landmark_type),
            'mask': None
        }

    # export mesh for debugging
    # added by Bony on 22-7-2019
    #lio.ply_from_array('mesh_before_nicp', id, mesh, '../output_dir/ply/', True, '__lsfm_masked')

    return return_dict
