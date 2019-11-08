import csv
import os
import numpy as np
from pathlib import Path
from functools import partial
from menpo.base import partial_doc, LazyList
from menpo.shape import ColouredTriMesh, TexturedTriMesh, TriMesh, PointCloud
import menpo.io as mio
from menpo.image.base import normalize_pixels_range
from menpo.io.output.base import _validate_filepath
from menpo.io.input import same_name, image_paths

export_pickle = partial(mio.export_pickle, protocol=2)
import_pickle = partial(mio.import_pickle, encoding='latin1')

def landmark_file_paths(pattern):
    r"""
    Return landmark file filepaths that Menpo3d can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, mesh_landmark_types)


same_name_landmark = partial_doc(same_name, paths_callable=landmark_file_paths)

def vtk_ensure_trilist(polydata):
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

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

def _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex):
    r"""
    Construct the correct Shape subclass given the inputs. TexturedTriMesh
    can only be created when tcoords and texture are available. ColouredTriMesh
    can only be created when colour_per_vertex is non None and TriMesh
    can only be created when trilist is non None. Worst case fall back is
    PointCloud.

    Parameters
    ----------
    points : ``(N, D)`` `ndarray`
        The N-D points.
    trilist : ``(N, 3)`` `ndarray`` or ``None``
        Triangle list or None.
    tcoords : ``(N, 2)`` `ndarray` or ``None``
        Texture coordinates.
    texture : :map:`Image` or ``None``
        Texture.
    colour_per_vertex : ``(N, 1)`` or ``(N, 3)`` `ndarray` or ``None``
        The colour per vertex.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
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


def obj_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

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
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

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

#function to write vertices vector into ply format, landmark points into pp file format (to be read by meshlab)
#written by Bony on 11-7-2019
def ply_from_array(filename, id, mesh, path, with_landmark, landmark_group, alignment):
    points = mesh.points
	#colours = mesh.colours
    faces = mesh.trilist

	#num_points = int(len(points)/3)
    filename_lm = path + 'landmarks/' + str(filename) + str(id) + '.pp'
    filename_ply = path + 'models/' + str(filename) + str(id) + '.ply'
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

        if with_landmark:
            header = '<!DOCTYPE PickedPoints> \n<PickedPoints> \n <DocumentData> \n  <DataFileName name="' + filename_ply + '"/> \n  <templateName name=""/> \n </DocumentData>\n'
            landmarks = mesh.landmarks._landmark_groups[landmark_group].points
            count = 0
            with open(filename_lm, 'w') as f:
                f.write(header)
                for points in landmarks:
                    count = count + 1
                    f.write('\t<point x="' + str(points[0]) + '" y="' + str(points[1]) + '" z="' + str(points[2]) + '" name="' + str(
                        count) + '" active="1"/>\n')
                f.write('</PickedPoints>')

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
    return [points_np, trilist_np]

def same_name_texture(path, paths_callable=image_paths):
    r"""
    Default image texture resolver. Returns **the lexicographically
    sorted first** texture found to have the same stem as the asset. A warning
    is raised if more than one texture is found.
    """
    # pattern finding all landmarks with the same stem
    pattern = path.with_suffix('.*')
    texture_paths = sorted(paths_callable(pattern))
    if len(texture_paths) > 1:
        warnings.warn('More than one texture found for file, returning '
                      'only the first.')
    if not texture_paths:
        return None
    return texture_paths[0]    

def import_mesh(path, hasTexture):
    if hasTexture:
        kwargs = {'texture_resolver': same_name_texture}
    else:
        kwargs = {}
    mesh = _import(path, mesh_types,                                      
                   importer_kwargs=kwargs)    
    if hasattr(mesh, 'texture'):
        if mesh.texture.pixels.dtype != np.float64:
            mesh.texture.pixels = normalize_pixels_range(mesh.texture.pixels)
    return mesh

def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return Path(os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(str(filepath))))))

def _possible_extensions_from_filepath(filepath):
    r"""
    Generate a list of possible extensions from the given filepath. Since
    filenames can contain '.' characters and some extensions are compound
    (e.g. '.pkl.gz'), there may be many possible extensions for a given
    path. Generate a list possible extensions, preferring longer extensions.

    Parameters
    ----------
    filepath : `Path`
        A pathlib Path.

    Returns
    -------
    possible_extensions : `list` of `str`
        A list of extensions **with** leading '.' characters and converted
        to lowercase.
    """
    suffixes = filepath.suffixes
    return [''.join(suffixes[i:]).lower() for i in range(len(suffixes))]    

def importer_for_filepath(filepath, extensions_map):
    r"""
    Given a filepath, return the appropriate importer as mapped by the
    extension map.

    Parameters
    ----------
    filepath : `pathlib.Path`
        The filepath to get importers for.
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        a subclass of :class:`Importer`. The extensions are expected to
        contain the leading period eg. `.obj`.

    Returns
    --------
    importer: :class:`menpo.io.base.Importer` instance
        Importer as found in the `extensions_map` instantiated for the
        filepath provided.
    """
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