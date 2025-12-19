__version__ = "2.5.0"

__short_description__ = ("Downsample images to 8-bit pixel art.",)
__license__ = "MIT"
__author__ = "Richard Nagyfi"
__github_username__ = "sedthh"

from .pal import Pal
from .pyx import Pyx
from .vid import Vid
from .backend import get_backend, is_gpu_available, CUPY_AVAILABLE
