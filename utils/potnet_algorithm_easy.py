import numpy as np
import warnings

def zeta(vecs, lattice_mat, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    """临时替代版本 - 避免Cython依赖"""
    warnings.warn("Using fallback PotNet implementation", UserWarning)
    
    if len(vecs.shape) == 1:
        vecs = vecs.reshape(1, -1)
    
    distances = np.linalg.norm(vecs, axis=1)
    distances = np.where(distances < 1e-10, 1e-10, distances)
    
    # 简化库仑势
    cutoff_factor = np.exp(-distances / R)
    potential = -param / distances * cutoff_factor
    potential = np.where(np.isfinite(potential), potential, 0.0)
    
    return potential

def exp(vecs, lattice_mat, param=3.0, R=3, eps=1e-12, parallel=False, verbose=False):
    """临时替代版本 - 泡利排斥势"""
    warnings.warn("Using fallback PotNet implementation", UserWarning)
    
    if len(vecs.shape) == 1:
        vecs = vecs.reshape(1, -1)
    
    distances = np.linalg.norm(vecs, axis=1)
    potential = param * np.exp(-distances)
    potential = np.where(np.isfinite(potential), potential, 0.0)
    
    return potential
