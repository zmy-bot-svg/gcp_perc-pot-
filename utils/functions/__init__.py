"""
PotNet函数模块
"""
try:
    from .series import (
        cython_upper_bessel,
        cython_upper_bessel_k, 
        cython_gsl_sf_gamma,
        cython_gsl_sf_gamma_inc
    )
    print("✅ PotNet Cython模块导入成功")
    POTNET_AVAILABLE = True
except ImportError as e:
    print(f"❌ PotNet Cython模块导入失败: {e}")
    print("请确保已经运行: python setup.py build_ext --inplace")
    POTNET_AVAILABLE = False

__all__ = ['cython_upper_bessel', 'cython_upper_bessel_k', 'cython_gsl_sf_gamma', 'cython_gsl_sf_gamma_inc', 'POTNET_AVAILABLE']
