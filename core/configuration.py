import numpy as np

class paramList:
    def __init__(self, LOD = 'low'):
        self.stiffness_weights = {
	  'low': lambda LOD: np.array([50, 20, 5, 2, .8, .5, .35, .2]), 
	  'mid': lambda LOD: np.array([50, 20, 8, 5, 3, 1]), 
          'high': lambda LOD: np.array([50, 20, 10, 8, 5])
        }[LOD](LOD)	
        self.landmark_weights = {
	  'low': lambda LOD: np.array([5, 2, .5, 0, 0, 0, 0, 0]), 
	  'mid': lambda LOD: np.array([5, 2, .5, 0, 0, 0]), 
          'high': lambda LOD: np.array([5, 2, 0.5, 0, 0])
        }[LOD](LOD)
        self.alignment = np.array([1000.0, -35.0, 1000.0, 57.0, 1000.0, 430.0])
        self.landmark_type = '__lsfm'
        self.template_ply = './template/template0.ply'
        self.template_ibug100 = './template/ibug100.pp'
        self.template_nosetip = './template/nosetip.pp'
        self.mode = "auto"
        self.ordering_method = "default"
