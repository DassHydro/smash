import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from . import core
from .core import *
from .core.model import *
from .core.utils import *

from . import solver
from .solver import *
from .solver.m_setup import *
from .solver.m_mesh import *
from .solver.m_input_data import *
from .solver.m_interface import *

from . import mesh
from .mesh import *
from .mesh.meshing import *

from . import io
from .io import *
from .io.yaml import *
from .io.hdf5 import *

__all__ = ["core", "solver", "mesh", "io"]
