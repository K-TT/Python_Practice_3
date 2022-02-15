Python 3.9.7 (v3.9.7:1016ef3790, Aug 30 2021, 16:39:15) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> iport numpy as np
SyntaxError: invalid syntax
>>> import numply as np
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    import numply as np
ModuleNotFoundError: No module named 'numply'
>>> import numpy as np
>>> from skimage import data
>>> import matplotlib.pyplot as plt
>>> camera=data.camera()
>>> camera
array([[200, 200, 200, ..., 189, 190, 190],
       [200, 199, 199, ..., 190, 190, 190],
       [199, 199, 199, ..., 190, 190, 190],
       ...,
       [ 25,  25,  27, ..., 139, 122, 147],
       [ 25,  25,  26, ..., 158, 141, 168],
       [ 25,  25,  27, ..., 151, 152, 149]], dtype=uint8)
>>> camera.dtype
dtype('uint8')
>>> camera.shape
(512, 512)
>>> moon=data.moon()
>>> moon
array([[116, 116, 122, ...,  93,  96,  96],
       [116, 116, 122, ...,  93,  96,  96],
       [116, 116, 122, ...,  93,  96,  96],
       ...,
       [109, 109, 112, ..., 117, 116, 116],
       [114, 114, 113, ..., 118, 118, 118],
       [114, 114, 113, ..., 118, 118, 118]], dtype=uint8)
>>> imporrt matplotlib.pyplot as plt
SyntaxError: invalid syntax
>>> import matplotlib.pyplot as plt
>>> plt.show()
>>> plt.imshow(camera,cmap='gray')
<matplotlib.image.AxesImage object at 0x7fc38e24cfa0>
>>> plt.show()
>>> from skimage import filters
pprint
>>> pprint.pprint(dir (filters))
Traceback (most recent call last):
  File "<pyshell#17>", line 1, in <module>
    pprint.pprint(dir (filters))
NameError: name 'pprint' is not defined
>>> import pprint
>>> pprint.pprint(dir (filters))
['LPIFilter2D',
 '__all__',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '_gabor',
 '_gaussian',
 '_guess_spatial_dimensions',
 '_median',
 '_multiotsu',
 '_rank_order',
 '_sparse',
 '_sparse_cy',
 '_unsharp_mask',
 '_window',
 'apply_hysteresis_threshold',
 'correlate_sparse',
 'difference_of_gaussians',
 'edges',
 'farid',
 'farid_h',
 'farid_v',
 'frangi',
 'gabor',
 'gabor_kernel',
 'gaussian',
 'hessian',
 'inverse',
 'laplace',
 'lpi_filter',
 'median',
 'meijering',
 'prewitt',
 'prewitt_h',
 'prewitt_v',
 'rank',
 'rank_order',
 'ridges',
 'roberts',
 'roberts_neg_diag',
 'roberts_pos_diag',
 'sato',
 'scharr',
 'scharr_h',
 'scharr_v',
 'sobel',
 'sobel_h',
 'sobel_v',
 'threshold_isodata',
 'threshold_li',
 'threshold_local',
 'threshold_mean',
 'threshold_minimum',
 'threshold_multiotsu',
 'threshold_niblack',
 'threshold_otsu',
 'threshold_sauvola',
 'threshold_triangle',
 'threshold_yen',
 'thresholding',
 'try_all_threshold',
 'unsharp_mask',
 'wiener',
 'window']
>>> filtered_images=filters.gaussian(camera,1)
>>> plt.imshow(filtered_images)
<matplotlib.image.AxesImage object at 0x7fc3910fcdc0>
>>> plt.show()
>>> plt.imshow(camera,cmap='gray')
<matplotlib.image.AxesImage object at 0x7fc38f12cdf0>
>>> plt.show()
>>> filtered_camera4=filters.gaussian(camera, 4)
>>> plt.imshow(filtered_camera4)
<matplotlib.image.AxesImage object at 0x7fc38f62e6a0>
>>> plt.show()
>>> 