import csv
import numpy as np
from menpo.shape import PointCloud, TriMesh
from menpo.transform import Translation, Scale
from core.io import getTriMeshfromPly
from core.configuration import paramList

# We define a template that we use for the landmarks
LANDMARK_MASK = np.ones(68, dtype=np.bool)
# Remove the jaw...
LANDMARK_MASK[:17] = False
# ...and the inner lips for robustness.
LANDMARK_MASK[-8:] = False

def prepare_template_reference_space(template):
    r"""Return a copy of the template centred at the origin
    and with max radial distance from centre of 1.

    This means the template is:
      1. fully contained by a bounding sphere of radius 1 at the origin
      2. centred at the origin.

    This isn't necessary, but it's nice to have a meaningful reference space
    for our models.
    """
    max_radial = np.sqrt(
        ((template.points - template.centre()) ** 2).sum(axis=1)).max()
    translation = Translation(-template.centre())
    scale = Scale(1 / max_radial, n_dims=3)
    adjustment = translation.compose_before(scale)

    adjustment.apply(template)
    return adjustment.apply(template)
		
def load_lanmarks_from_mesh(mesh):
    #to obtain landmark
    param = paramList()
    landmark_68 = []
    landmark_100 = []
    landmark_ear = []
    nosetip = []
    count = 0
    with open(param.template_ibug100) as pp_file:
        pp_file = csv.reader(pp_file, delimiter='"')
        for row in pp_file:
            count = count + 1
            if count >= 7 and count < 107:
                landmark_100.append([float(row[7]), float(row[9]), float(row[5])])
                if count < 75:
                    landmark_68.append([float(row[7]), float(row[9]), float(row[5])])
            if count >= 84 and count <= 94:
                landmark_ear.append([float(row[7]), float(row[9]), float(row[5])])
            if count >= 96 and count <= 106:
                landmark_ear.append([float(row[7]), float(row[9]), float(row[5])])
    count = 0
    with open(param.template_nosetip) as pp_file:
        pp_file = csv.reader(pp_file, delimiter='"')
        for row in pp_file:
            count = count + 1
            if count >= 7 and count < 8:
                nosetip.append([float(row[7]), float(row[9]), float(row[5])])
    mesh.landmarks['ibug68'] = PointCloud(np.array(landmark_68))
    mesh.landmarks['ibug100'] = PointCloud(np.array(landmark_100))
    mesh.landmarks['ibugEar'] = PointCloud(np.array(landmark_ear))
    mesh.landmarks['nosetip'] = PointCloud(np.array(nosetip))
    mesh.landmarks['__lsfm'] = mesh.landmarks['ibug68'].lms.from_mask(
        LANDMARK_MASK)
    return prepare_template_reference_space(mesh)

def mesh_polish(mesh, lm_filename, landmark_type='ibug68'):
    rescaled_mesh = mesh
    param = paramList()
    scale = param.alignment[0]
    offset = [param.alignment[1], param.alignment[3], param.alignment[5]]
    for i in range(len(mesh.points)):
        rescaled_mesh.points[i][0] = mesh.points[i][0] * scale + offset[0]
        rescaled_mesh.points[i][1] = mesh.points[i][1] * (-1) * scale + offset[1]
        rescaled_mesh.points[i][2] = mesh.points[i][2] * (-1) * scale + offset[2]
    landmark_unre = []
    count = 0
    with open(lm_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            if landmark_type == '__lsfm':
                if count >= 17 and count <= 59:
                    landmark_unre.append([float(row[0]) * scale + offset[0], float(row[1]) * (-1) * scale + offset[1], float(row[2]) * (-1) * scale + offset[2]])
            if landmark_type == 'ibug68':
                if count < 68:
                    landmark_unre.append([float(row[0]) * scale + offset[0], float(row[1]) * (-1) * scale + offset[1], float(row[2]) * (-1) * scale + offset[2]])
            count = count + 1
    rescaled_mesh.landmarks['__lsfm'] = PointCloud(np.array(landmark_unre))
    rescaled_mesh.landmarks['unre_pickpoints'] = PointCloud(np.array(landmark_unre))
    return rescaled_mesh
