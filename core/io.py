import csv
import os
import numpy as np
from pathlib import Path
from menpo.shape import PointCloud, TriMesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy 

def _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex):
    # Four different outcomes - either we have a textured mesh, a coloured
    # mesh or just a plain mesh or we fall back to a plain pointcloud.
    if trilist is None:
        obj = PointCloud(points, copy=False)
    elif tcoords is not None and texture is not None:
        obj = TexturedTriMesh(points, tcoords, texture,
                              trilist=trilist, copy=False)
    elif colour_per_vertex is not None:
        obj = ColouredTriMesh(points, trilist=trilist,
                              colours=colour_per_vertex, copy=False)
    else:
        # TriMesh fall through
        obj = TriMesh(points, trilist=trilist, copy=False)

    if tcoords is not None and texture is None:
        warnings.warn('tcoords were found, but no texture was recovered, '
                      'reverting to an untextured mesh.')
    if texture is not None and tcoords is None:
        warnings.warn('texture was found, but no tcoords were recovered, '
                      'reverting to an untextured mesh.')

    return obj

def vtk_ensure_trilist(polydata):
    try:
        trilist = vtk_to_numpy(polydata.GetPolys().GetData())

        # 5 is the triangle type - if we have another type we need to
        # use a vtkTriangleFilter
        c = vtk.vtkCellTypes()
        polydata.GetCellTypes(c)

        if c.GetNumberOfTypes() != 1 or polydata.GetCellType(0) != 5:
            warnings.warn('Non-triangular mesh connectivity was detected - '
                          'this is currently unsupported and thus the '
                          'connectivity is being coerced into a triangular '
                          'mesh. This may have unintended consequences.')
            t_filter = vtk.vtkTriangleFilter()
            t_filter.SetInputData(polydata)
            t_filter.Update()
            trilist = vtk_to_numpy(t_filter.GetOutput().GetPolys().GetData())

        return trilist.reshape([-1, 4])[:, 1:]
    except Exception as e:
        warnings.warn(str(e))
        return None

def obj_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    obj_importer = vtk.vtkOBJReader()
    obj_importer.SetFileName(str(filepath))
    obj_importer.Update()

    # Get the output
    polydata = obj_importer.GetOutput()

    # We must have point data!
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

    trilist = np.require(vtk_ensure_trilist(polydata), requirements=['C'])

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = None
    if texture is not None:
        try:
            tcoords = vtk_to_numpy(polydata.GetPointData().GetTCoords())
        except Exception:
            pass

        if isinstance(tcoords, np.ndarray) and tcoords.size == 0:
            tcoords = None

    colour_per_vertex = None
    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)


def ply_importer(filepath, asset=None, texture_resolver=None, **kwargs):    
    ply_importer = vtk.vtkPLYReader()
    ply_importer.SetFileName(str(filepath))

    ply_importer.Update()

    # Get the output
    polydata = ply_importer.GetOutput()

    # We must have point data!
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

    trilist = np.require(vtk_ensure_trilist(polydata), requirements=['C'])

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = None
    if texture is not None:
        try:
            tcoords = vtk_to_numpy(polydata.GetPointData().GetTCoords())
        except Exception:
            pass

        if isinstance(tcoords, np.ndarray) and tcoords.size == 0:
            tcoords = None

    colour_per_vertex = None
    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)  

mesh_types = {'.obj': obj_importer,              
              '.ply': ply_importer }
			  
def import_mesh(path):
    kwargs = {}
    mesh = _import(path, mesh_types,                                      
                   importer_kwargs=kwargs)  
    #if hasattr(mesh, 'texture'):
    #    if mesh.texture.pixels.dtype != np.float64:
    #        mesh.texture.pixels = normalize_pixels_range(mesh.texture.pixels)
    return mesh

def _norm_path(filepath):
    return Path(os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(str(filepath))))))

def _possible_extensions_from_filepath(filepath):
    suffixes = filepath.suffixes
    return [''.join(suffixes[i:]).lower() for i in range(len(suffixes))]  

def getTriMeshfromPly(path):
    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data.append(row)

    flag = False
    points = []
    trilist = []
    count = 0
    for row in range(len(data)):
        if (data[row][0] == 'element') and (data[row][1] == 'vertex'):
            numOfVertices = int(data[row][2])
        if flag and count < numOfVertices:
            data[row][0] = "{0:.6f}".format(float(data[row][0]))
            data[row][1] = "{0:.6f}".format(float(data[row][1]))
            data[row][2] = "{0:.6f}".format(float(data[row][2]))
            points.append([float(data[row][0]), float(data[row][1]), float(data[row][2])])
            count = count + 1
        elif flag and count >= numOfVertices:
            if data[row][0] == '3':
                trilist.append([int(data[row][1]), int(data[row][2]), int(data[row][3])])
        if (data[row][0] == 'end_header'):
            flag = True
    points_np = np.array(points)
    trilist_np = np.array(trilist)
    return TriMesh(points_np, trilist_np)
	
#function to write vertices vector into ply format, landmark points into pp file format (to be read by meshlab)
#written by Bony on 11-7-2019
def ply_from_array(filename, mesh, with_landmark, landmark_group, alignment=[ 1, 0, 1, 0, 1, 0 ], vid=None):
    points = mesh.points
    #colours = mesh.colours
    faces = mesh.trilist

    #num_points = int(len(points)/3)
    filename_lm = filename + '.pp'
    filename_ply = filename
    header = '''ply
format ascii 1.0
comment UNRE generated
element vertex {0}
property float x
property float y
property float z
element face {1}
property list uchar int vertex_indices
end_header\n'''.format(mesh.n_points, mesh.n_tris)

    vertice_list=[]
    colours_list=[]
    faces_list=[]
    for item in points:
        vertice_list.append(item)
    #for item in colours:
    #	colours_list.append(item)
    for item in faces:
        faces_list.append(item)

    with open(filename_ply, 'w') as f:
        f.writelines(header)
        for idx in range(0, mesh.n_points):
            for i in range(0,3):
                f.write(str((vertice_list[idx][i] - alignment[i*2+1]) * alignment[i*2]))
                f.write(' ')
            f.write('\n')

        for idx in range(0, mesh.n_tris):
            f.write('3 ')
            f.write(str(int(faces_list[idx][0])))
            f.write(' ')
            f.write(str(int(faces_list[idx][1])))
            f.write(' ')
            f.write(str(int(faces_list[idx][2])))
            f.write(' ')
            f.write('\n')

        if with_landmark and vid is None:        
            header = '<!DOCTYPE PickedPoints> \n<PickedPoints> \n <DocumentData> \n  <DataFileName name="' + filename_ply + '"/> \n  <templateName name=""/> \n </DocumentData>\n'
            landmarks = mesh.landmarks[landmark_group].points
            count = 0
            with open(filename_lm, 'w') as f:
                f.write(header)
                for points in landmarks:
                    count = count + 1
                    f.write('\t<point x="' + str((points[0] - alignment[1]) * alignment[0])
				+ '" y="' + str((points[1] - alignment[3]) * alignment[2])
				+ '" z="' + str((points[2] - alignment[5]) * alignment[4]) 
				+ '" name="' + str(count) + '" active="1"/>\n')
                f.write('</PickedPoints>')

        if with_landmark and vid is not None:
            header = '<!DOCTYPE PickedPoints> \n<PickedPoints> \n <DocumentData> \n  <DataFileName name="' + filename_ply + '"/> \n  <templateName name=""/> \n </DocumentData>\n'        
            count = 0
            with open(filename_lm, 'w') as f:
                f.write(header)
                for vertex_id in vid:
                    count = count + 1
                    f.write('\t<point x="' + str((vertice_list[vertex_id][0] - alignment[1]) * alignment[0])
				+ '" y="' + str((vertice_list[vertex_id][1] - alignment[3]) * alignment[2])
				+ '" z="' + str((vertice_list[vertex_id][2] - alignment[5]) * alignment[4]) 
				+ '" name="' + str(count) + '" active="1"/>\n')
                f.write('</PickedPoints>')

def importer_for_filepath(filepath, extensions_map):
    possible_exts = _possible_extensions_from_filepath(filepath)

    # we couldn't find an importer for all the suffixes (e.g .foo.bar)
    # maybe the file stem has '.' in it? -> try again but this time just use the
    # final suffix (.bar). (Note we first try '.foo.bar' as we want to catch
    # cases like '.pkl.gz')
    importer_callable = None
    while importer_callable is None and possible_exts:
        importer_callable = extensions_map.get(possible_exts.pop(0))

    if importer_callable is None:
        raise ValueError("{} does not have a "
                         "suitable importer.".format(filepath.name))
    return importer_callable
				
def _import(filepath, extensions_map,            
            asset=None, importer_kwargs=None):    
    path = _norm_path(filepath)
    if not path.is_file():
        raise ValueError("{} is not a file".format(path))

    # below could raise ValueError as well...
    importer_callable = importer_for_filepath(path, extensions_map)
    if importer_kwargs is None:
        importer_kwargs = {}
    built_objects = importer_callable(path, asset=asset, **importer_kwargs)

    # landmarks are iterable so check for list precisely
    if not isinstance(built_objects, list):
        built_objects = [built_objects]

    # attach path if there is no x.path already.
    for x in built_objects:
        if not hasattr(x, 'path'):
            try:
                x.path = path
            except AttributeError:
                pass  # that's fine! Probably a dict/list from PickleImporter.

    if len(built_objects) == 1:
        built_objects = built_objects[0]

    return built_objects
