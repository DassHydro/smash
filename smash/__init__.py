import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from . import core
from .core import *
from .core.model import *

from . import solver
from .solver import *
from .solver.m_setup import *
from .solver.m_mesh import *
from .solver.m_input_data import *

from . import meshing
from .meshing import *

from . import io
from .io import *

__all__ = ["core", "solver", "meshing", "io"]
