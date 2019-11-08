
from menpo.shape import ColouredTriMesh
from menpo.shape  import PointCloud
from menpo.transform import Translation, Scale
import os_io as lio
from pathlib import Path
import csv
import numpy as np
from functools import lru_cache

DATA_DIR = '/home/u/workspace/VPE/template'
# We define a template that we use for the landmarks
LANDMARK_MASK = np.ones(68, dtype=np.bool)
# Remove the jaw...
LANDMARK_MASK[:17] = False
# ...and the inner lips for robustness.
LANDMARK_MASK[-8:] = False

def path_to_template():
    return (DATA_DIR + '/template.pkl')

def save_template(template, overwrite=False):
    lio.export_pickle(template, path_to_template(), overwrite=overwrite)

def load_mean_from_basel(path):
    mm = loadmat(str(path))
    trilist = mm['tl'][:, [0, 2, 1]] - 1
    mean_points = mm['shapeMU'].reshape(-1, 3)
    mean_colour = mm['texMU'].reshape(-1, 3) / 255
    return ColouredTriMesh(mean_points, trilist=trilist, colours=mean_colour)


def load_basel_template_metadata():
    return lio.import_pickle(DATA_DIR + '/basel_template_metadata.pkl')


def generate_template_from_basel_and_metadata(basel, meta):
    template = ColouredTriMesh(basel.points[meta['map_tddfa_to_basel']],
                               trilist=meta['tddfa_trilist'],
                               colours=basel.colours[
                                   meta['map_tddfa_to_basel']])

    template.landmarks['ibug68'] = meta['landmarks']['ibug68']
    template.landmarks['nosetip'] = meta['landmarks']['nosetip']

    return template

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

#@lru_cache()
def load_template():
    template = lio.import_pickle(DATA_DIR + '/template.pkl')
    # reorder the trilist so as to invert the normals
    # modified by Bony on 14-8-2019
    #for idx in range(0, len(template.trilist)):
    #    trilist_0 = template.trilist[idx][0]
    #    trilist_1 = template.trilist[idx][1]
    #    trilist_2 = template.trilist[idx][2]
    #    template.trilist[idx] = [trilist_0, trilist_2, trilist_1]

    template.landmarks['__lsfm'] = template.landmarks['ibug68'].lms.from_mask(
        LANDMARK_MASK)
    return prepare_template_reference_space(template)

def save_template_from_mesh(path):
    points, trilist = lio.getTriMeshfromPly(path)
    template = ColouredTriMesh(points, trilist)
    #to obtain landmark
    landmark_68 = []
    landmark_100 = []
    landmark_ear = []
    nosetip = []
    count = 0
    with open('/home/u/workspace/VPE/template/ibug100.pp') as pp_file:
        pp_file = csv.reader(pp_file, delimiter='"')
        for row in pp_file:
            count = count + 1
            if count >= 7 and count < 107:
                landmark_100.append([float(row[1]), float(row[3]), float(row[5])])
                if count < 75:
                    landmark_68.append([float(row[1]), float(row[3]), float(row[5])])
            if count >= 84 and count <= 94:
                landmark_ear.append([float(row[1]), float(row[3]), float(row[5])])
            if count >= 96 and count <= 106:
                landmark_ear.append([float(row[1]), float(row[3]), float(row[5])])
    count = 0
    with open('/home/u/workspace/VPE/template/nosetip.pp') as pp_file:
        pp_file = csv.reader(pp_file, delimiter='"')
        for row in pp_file:
            count = count + 1
            if count >= 7 and count < 8:
                nosetip.append([float(row[1]), float(row[3]), float(row[5])])
    template.landmarks['ibug68'] = PointCloud(np.array(landmark_68))
    template.landmarks['ibug100'] = PointCloud(np.array(landmark_100))
    template.landmarks['ibugEar'] = PointCloud(np.array(landmark_ear))
    template.landmarks['nosetip'] = PointCloud(np.array(nosetip))
    save_template(template, overwrite=True)
