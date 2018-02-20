import sys
from . import layers
from . import layers as nn
from . import utils
from . import nputils
from . import tfutils
from . import distributions
from .utils import FileWriter
from .tfutils import function, TensorDict

if 'ipykernel' in sys.argv[0]:
    from . import nbutils

__version__ = '0.3.0'
