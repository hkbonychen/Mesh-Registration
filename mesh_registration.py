import sys, time, csv, sys, os
import numpy as np
from core.base import load_lanmarks_from_mesh, mesh_polish
from core.io import import_mesh, getTriMeshfromPly, ply_from_array
from core import landmark_and_correspond_mesh
from core.configuration import paramList

os.environ['CHOLMOD_USE_GPU'] = '1'

def main():
    if not len(sys.argv) == 5:
        print('Invalid input arguments: mesh_registration <input obj file path> <input landmark file path> <output ply file path> <level of detail: "low", "mid", "high">')
        return
        
    start = time.time()	
    param = paramList()
    landmark_type = param.landmark_type
    template = load_lanmarks_from_mesh(getTriMeshfromPly(param.template_ply))
    try:
        mesh = import_mesh(sys.argv[1])
        mesh = mesh_polish(mesh, sys.argv[2], landmark_type = landmark_type)
        c = landmark_and_correspond_mesh(template, mesh, verbose=False, landmark_type=landmark_type,
                                          LOD = sys.argv[4])
    except Exception as e:
        if type(e) is KeyboardInterrupt:
            print(e)
            sys.exit(1)
        print('{} - FAILED TO CORRESPOND: {}'.format(0, e))
        return
    #write the output to ply
    alignment = [ param.alignment[0] ** -1, param.alignment[1], 
                    (-1) * param.alignment[2] ** -1, param.alignment[3], 
                    (-1) * param.alignment[4] ** -1, param.alignment[5] ]
    ply_from_array(sys.argv[3], c['shape_nicp'], with_landmark = False, landmark_group = landmark_type, alignment = alignment)
    end = time.time()
    print(end-start)
	
if __name__ == "__main__":
    main()
