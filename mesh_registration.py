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
        
    overall_start = time.time()	
    param = paramList()
    landmark_type = param.landmark_type
    if param.profile_time:
        load_template_start = time.time()
    template, vid = load_lanmarks_from_mesh(getTriMeshfromPly(param.template_ply))    
    ply_from_array('./output/template.ply', template, with_landmark = True, landmark_group = 'ibug100', vid = None)
    if param.profile_time:
        print("load_template time elapsed: " + str(time.time()-load_template_start))
    try:
        if param.profile_time:
            load_mesh_start = time.time()
        mesh = import_mesh(sys.argv[1])
        mesh = mesh_polish(mesh, sys.argv[2], landmark_type = landmark_type)
        if param.profile_time:
            print("load input mesh time elapsed: " + str(time.time()-load_mesh_start))
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
    if param.profile_time:
        ply_output_start = time.time()
    ply_from_array(sys.argv[3], c['shape_nicp'], with_landmark = True, landmark_group = 'ibug100', alignment = alignment, vid = None)
    if param.profile_time:
        print("ply output elapsed: " + str(time.time()-ply_output_start))
    print("overall time elapsed: " + str(time.time()-overall_start))
	
if __name__ == "__main__":
    main()
