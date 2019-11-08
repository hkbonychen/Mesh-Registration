import sys
import time
import csv
import sys
import numpy as np
from menpo.shape import PointCloud

#sys.path.insert(0, '/home/u/workspace/one-shot') 
import os_io as lio
from core import landmark_and_correspond_mesh
from basel import save_template_from_mesh, load_template

def polish(mesh, lm_filename, landmark_type='ibug68'):
    scale = 1000
    x_offset = -35
    y_offset = 57
    z_offset = 430
    rescaled_mesh = mesh
    for i in range(len(mesh.points)):
        rescaled_mesh.points[i][0] = mesh.points[i][0] * scale + x_offset
        rescaled_mesh.points[i][1] = mesh.points[i][1] * (-1) * scale + y_offset
        rescaled_mesh.points[i][2] = mesh.points[i][2] * (-1) * scale + z_offset

    landmark_unre = []        
    count = 0
    with open(lm_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            if landmark_type == '__lsfm':                        
                if count >= 17 and count <= 59:
                    landmark_unre.append([float(row[0]) * 1000 - 35, float(row[1]) * -1000 + 57, float(row[2]) * -1000 + 430])
            if landmark_type == 'ibug68':
                if count < 68:
                    landmark_unre.append([float(row[0]) * 1000 - 35, float(row[1]) * -1000 + 57, float(row[2]) * -1000 + 430])
            count = count + 1
    rescaled_mesh.landmarks['unre_pickpoints'] = PointCloud(np.array(landmark_unre))
    return rescaled_mesh

def main():
    start = time.time()
    save_template_from_mesh('/home/u/workspace/VPE/template/template0.ply')
    template = load_template().copy()
    alignment = [1, 0, 1, 0, 1, 0]
    #lio.ply_from_array('template', 0, template, '/home/u/workspace/VPE/output/', True, 'ibug68', alignment)        
    try:
        mesh = lio.import_mesh(sys.argv[1], hasTexture=False)
        mesh = polish(mesh, sys.argv[2], landmark_type='__lsfm')
        c = landmark_and_correspond_mesh(template, mesh, verbose=True, landmark_type='__lsfm')
    except Exception as e:
        if type(e) is KeyboardInterrupt:
            print(e)
            sys.exit(1)
        print('{} - FAILED TO CORRESPOND: {}'.format(0, e))
        return
    #lio.export_shape_nicp(r, id_, c['shape_nicp'])
    alignment = [ 1/1000, -35, -1 / 1000, 57, -1 / 1000, 430 ]
    lio.ply_from_array('', 0, c['shape_nicp'], '/home/u/workspace/VPE/output/', True, '__lsfm', alignment)
    end = time.time()
    print(end - start)
    
if __name__ == "__main__":
    main()
