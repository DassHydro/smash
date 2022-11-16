from __future__ import annotations

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mw_forward import forward

from smash.core._build_model import (
    _parse_derived_type,
    _build_setup,
    _build_mesh,
    _build_input_data,
)

from smash.core._optimize import (
    _standardize_optimize_args,
    _standardize_optimize_options,
    _optimize_sbs,
    _optimize_lbfgsb,
    _optimize_nelder_mead,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.
    **S**\patially distributed **M**\odelling and **AS**\simillation for **H**\ydrology.

    Parameters
    ----------
    setup : dict
        Model initialization setup dictionary (see: :ref:`setup arguments <user_guide.model_initialization.setup>`).

    mesh : dict
        Model initialization mesh dictionary. (see: :ref:`mesh arguments <user_guide.model_initialization.mesh>`).

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization

    See Also
    --------
    generate_mesh: Automatic mesh generation.
    """

    def __init__(self, setup: dict, mesh: dict):

        if setup or mesh:

            if isinstance(setup, dict):

                descriptor_name = setup.get("descriptor_name", [])

                nd = 1 if isinstance(descriptor_name, str) else len(descriptor_name)

                self.setup = SetupDT(nd, mesh["ng"])

                _parse_derived_type(self.setup, setup)

            else:

                raise TypeError(
                    f"'setup' argument must be dictionary, not {type(setup)}"
                )

            _build_setup(self.setup)

            if isinstance(mesh, dict):

                self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

                _parse_derived_type(self.mesh, mesh)

            else:
                raise TypeError(f"'mesh' argument must be dictionary, not {type(mesh)}")

            _build_mesh(self.setup, self.mesh)

            self.input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self.input_data)

            self.parameters = ParametersDT(self.mesh)

            self.states = StatesDT(self.mesh)

            self.output = OutputDT(self.setup, self.mesh)

            self._last_update = "Initialization"

    def __repr__(self):

        structure = f"Structure: '{self.setup.structure}'"
        dim = f"Spatio-Temporal dimension: (x: {self.mesh.ncol}, y: {self.mesh.nrow}, time: {self.setup._ntime_step})"
        last_update = f"Last update: {self._last_update}"

        return f"{structure}\n{dim}\n{last_update}"

    @property
    def setup(self):

        """
        The setup of the Model.

        The model setup is represented as a SetupDT object. See `SetupDT <smash.solver._mwd_setup.SetupDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.setup.<TAB>
        model.setup.copy(                   model.setup.prcp_directory
        model.setup.daily_interannual_pet   model.setup.prcp_format
        model.setup.descriptor_directory    model.setup.prcp_indice
        model.setup.descriptor_format       model.setup.qobs_directory
        model.setup.descriptor_name         model.setup.read_descriptor
        model.setup.dt                      model.setup.read_pet
        model.setup.end_time                model.setup.read_prcp
        model.setup.from_handle(            model.setup.read_qobs
        model.setup.mean_forcing            model.setup.save_net_prcp_domain
        model.setup.pet_conversion_factor   model.setup.save_qsim_domain
        model.setup.pet_directory           model.setup.sparse_storage
        model.setup.pet_format              model.setup.start_time
        model.setup.prcp_conversion_factor  model.setup.structure

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """

        return self._setup

    @setup.setter
    def setup(self, value):

        if isinstance(value, SetupDT):
            self._setup = value

        else:
            raise TypeError(
                f"'setup' attribute must be set with {type(SetupDT())}, not {type(value)}"
            )

    @property
    def mesh(self):
        """
        The mesh of the Model.

        The model mesh is represented as a MeshDT object. See `MeshDT <smash.solver._mwd_mesh.MeshDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.mesh.<TAB>
        model.mesh.active_cell   model.mesh.gauge_pos
        model.mesh.area          model.mesh.nac
        model.mesh.code          model.mesh.ncol
        model.mesh.copy(         model.mesh.ng
        model.mesh.drained_area  model.mesh.nrow
        model.mesh.dx            model.mesh.path
        model.mesh.flwdir        model.mesh.xmin
        model.mesh.flwdst        model.mesh.ymax
        model.mesh.from_handle(

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """

        return self._mesh

    @mesh.setter
    def mesh(self, value):

        if isinstance(value, MeshDT):
            self._mesh = value

        else:
            raise TypeError(
                f"'mesh' attribute must be set with {type(MeshDT())}, not {type(value)}"
            )

    @property
    def input_data(self):
        """
        The input data of the Model.

        The model input data is represented as a Input_DataDT object. See `Input_DataDT <smash.solver._mwd_input_data.Input_DataDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.input_data.<TAB>
        model.input_data.copy(         model.input_data.prcp
        model.input_data.descriptor    model.input_data.prcp_indice
        model.input_data.from_handle(  model.input_data.qobs
        model.input_data.mean_pet      model.input_data.sparse_pet
        model.input_data.mean_prcp     model.input_data.sparse_prcp
        model.input_data.pet

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """
        return self._input_data

    @input_data.setter
    def input_data(self, value):

        if isinstance(value, Input_DataDT):
            self._input_data = value

        else:
            raise TypeError(
                f"'input_data' attribute must be set with {type(Input_DataDT())}, not {type(value)}"
            )

    @property
    def parameters(self):
        """
        The parameters of the Model.

        The model parameters is represented as a ParametersDT object. See `ParametersDT <smash.solver._mwd_parameters.ParametersDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.parameters.<TAB>
        model.parameters.alpha         model.parameters.cusl1
        model.parameters.b             model.parameters.cusl2
        model.parameters.beta          model.parameters.ds
        model.parameters.cft           model.parameters.dsm
        model.parameters.ci            model.parameters.exc
        model.parameters.clsl          model.parameters.from_handle(
        model.parameters.copy(         model.parameters.ks
        model.parameters.cp            model.parameters.lr
        model.parameters.cst           model.parameters.ws

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):

        if isinstance(value, ParametersDT):

            self._parameters = value

        else:
            raise TypeError(
                f"'parameters' attribute must be set with {type(ParametersDT())}, not {type(value)}"
            )

    @property
    def states(self):
        """
        The states of the Model.

        The model states is represented as a StatesDT object. See `StatesDT <smash.solver._mwd_states.StatesDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.states.<TAB>
        model.states.copy(         model.states.hlsl
        model.states.from_handle(  model.states.hp
        model.states.hft           model.states.hst
        model.states.hi            model.states.husl1
        model.states.hlr           model.states.husl2

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """

        return self._states

    @states.setter
    def states(self, value):

        if isinstance(value, StatesDT):

            self._states = value

        else:
            raise TypeError(
                f"'states' attribute must be set with {type(StatesDT())}, not {type(value)}"
            )

    @property
    def output(self):
        """
        The output of the Model.

        The model output is represented as a OutputDT object. See `OutputDT <smash.solver._mwd_output.OutputDT>`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.output.<TAB>
        model.output.an                   model.output.parameters_gradient
        model.output.copy(                model.output.qsim
        model.output.cost                 model.output.qsim_domain
        model.output.from_handle(         model.output.sp1
        model.output.fstates              model.output.sp2
        model.output.ian                  model.output.sparse_qsim_domain

        Notes
        -----
        This object is a wrapped derived type from `f90wrap <https://github.com/jameskermode/f90wrap>`__.
        """

        return self._output

    @output.setter
    def output(self, value):

        if isinstance(value, OutputDT):

            self._output = value

        else:
            raise TypeError(
                f"'output' attribute must be set with {type(OutputDT())}, not {type(value)}"
            )

    @property
    def _last_update(self):

        return self.__last_update

    @_last_update.setter
    def _last_update(self, value):

        if isinstance(value, str):

            self.__last_update = value

        else:
            raise TypeError(
                f"'_last_update' attribute must be set with {str}, not {type(value)}"
            )

    def copy(self):
        """
        Make a deepcopy of the Model.

        Returns
        -------
        Model
            A copy of Model.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Create a pointer towards Model

        >>> model_ptr = model
        >>> model_ptr.parameters.cp = 1
        >>> model_ptr.parameters.cp[0,0], model.parameters.cp[0,0]
        (1.0, 1.0)

        Create a deepcopy of Model

        >>> model_dc = model.copy()
        >>> model_dc.parameters.cp = 200
        >>> model_dc.parameters.cp[0,0], model.parameters.cp[0,0]
        (200.0, 1.0)
        """

        copy = Model(None, None)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy.input_data = self.input_data.copy()
        copy.parameters = self.parameters.copy()
        copy.states = self.states.copy()
        copy.output = self.output.copy()
        copy._last_update = self._last_update

        return copy

    def run(self, inplace: bool = False):
        """
        Run the Model.

        Parameters
        ----------
        inplace : bool, default False
            if True, perform operation in-place.

        Returns
        -------
        Model : Model or None
            Model with run outputs or None if inplace.

        Notes
        -----
        This method is directly calling the forward model :math:`Y = M(k)`.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> model.run(inplace=True)
        >>> model
        Structure: 'gr-a'
        Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
        Last update: Forward Run

        Access to simulated discharge

        >>> model.output.qsim[0,:]
        array([1.9826449e-03, 1.3466686e-07, 6.7618025e-12, ..., 2.0916510e+01,
               2.0762346e+01, 2.0610489e+01], dtype=float32)
        """

        if inplace:

            instance = self

        else:

            instance = self.copy()

        print("</> Forward Model Y = M (k)")

        cost = np.float32(0)

        forward(
            instance.setup,
            instance.mesh,
            instance.input_data,
            instance.parameters,
            instance.parameters.copy(),
            instance.states,
            instance.states.copy(),
            instance.output,
            cost,
        )

        instance._last_update = "Forward Run"

        if not inplace:

            return instance

    def optimize(
        self,
        mapping: str = "uniform",
        algorithm: str | None = None,
        control_vector: str | list | tuple | set | None = None,
        jobs_fun: str = "nse",
        bounds: list | tuple | set | None = None,
        gauge: str | list | tuple | set = "downstream",
        wgauge: str | list | tuple | set = "mean",
        ost: str | pd.Timestamp | None = None,
        options: dict | None = None,
        inplace: bool = False,
    ):
        """
        Optimize the Model.

        .. hint::
            See the :ref:`user_guide` for more.

        Parameters
        ----------
        mapping : str, default 'uniform'
            Type of mapping. Should be one of

            - 'uniform'
            - 'distributed'
            - 'hyper-linear'
            - 'hyper-polynomial'

        algorithm : str or None, default None
            Type of algorithm. Should be one of

            - 'sbs'
            - 'nelder-mead'
            - 'l-bfgs-b'

            .. note::
                If not given, chosen to be one of ``sbs`` or ``l-bfgs-b`` depending on the optimization mapping.

        control_vector : str, sequence or None, default None
            Parameters and/or states to be optimized. The control vector argument
            can be any parameter or state name or any sequence of parameter and/or state names.

            .. note::
                If not given, the control vector will be composed of the parameters of the structure defined in the Model setup.

        jobs_fun : str, default 'nse'
            Type of objective function to be minimized. Should be one of

            - 'nse'
            - 'kge'
            - 'kge2'
            - 'se'
            - 'rmse'
            - 'logarithmic'

        bounds : sequence or None, default None
            Bounds on control vector. The bounds argument is a sequence of ``(min, max)``.
            The size of the bounds sequence must be equal to the control vector size.
            The bounds argument accepts pairs of values with ``min`` lower than ``max``.
            None value inside the sequence will be filled in with default bound values.

            .. note::
                If not given, the bounds will be filled in with default bound values.

        gauge : str, sequence, default 'downstream'
            Type of gauge to be optimized. There are two ways to specify it:

            1. A gauge code or any sequence of gauge codes.
               The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
            2. An alias among ``all`` and ``downstream``. ``all`` is equivalent to a sequence of all gauge codes.
               ``downstream`` is equivalent to the gauge code of the most downstream gauge.

        wgauge : str, sequence, default 'mean'
            Type of gauge weights. There are two ways to specify it:

            1. A sequence of value whose size must be equal to the number of gauges optimized.
            2. An alias among ``mean``, ``area`` or ``minv_area``.

        ost : str, pandas.Timestamp or None, default None
            The optimization start time. The optimization will only be performed between the
            optimization start time ``ost`` and the end time. The value can be a str which can be interpreted by
            pandas.Timestamp `(see here) <https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html>`__.
            The ``ost`` date value must be between the start time and the end time defined in the Model setup.

            .. note::
                If not given, the optimization start time will be equal to the start time.

        options : dict or None, default None
            A dictionary of algorithm options.

        inplace : bool, default False
            if True, perform operation in-place.

        Returns
        -------
        Model : Model or None
            Model with optimize outputs or None if inplace.

        Notes
        -----
        This method is directly calling the forward model :math:`Y = M(k)` and the adjoint model
        :math:`\delta k^* = \\left( \\frac{\delta M}{\delta k} \\right)^* . \delta Y^*`
        if the algorithm ``l-bfgs-b`` is choosen to retrieve the gradient of the cost function wrt the control vector.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> model.optimize(inplace=True)
        >>> model
        Structure: 'gr-a'
        Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
        Last update: Step By Step Optimization

        Access to simulated discharge

        >>> model.output.qsim[0,:]
        array([5.7140866e-04, 4.7018618e-04, 3.5345653e-04, ..., 1.9009293e+01,
               1.8772749e+01, 1.8541389e+01], dtype=float32)

        Access to optimized parameters

        >>> ind = tuple(model.mesh.gauge_pos[0,:])
        >>> ind
        (20, 27)
        >>> (
        ... "cp", model.parameters.cp[ind],
        ... "cft", model.parameters.cft[ind],
        ... "exc", model.parameters.exc[ind],
        ... "lr", model.parameters.lr[ind],
        ... )
        ('cp', 76.57858, 'cft', 263.64627, 'exc', -1.4613823, 'lr', 30.859276)
        """

        if inplace:

            instance = self

        else:

            instance = self.copy()

        (
            mapping,
            algorithm,
            control_vector,
            jobs_fun,
            bounds,
            wgauge,
            ost,
        ) = _standardize_optimize_args(
            mapping,
            algorithm,
            control_vector,
            jobs_fun,
            bounds,
            gauge,
            wgauge,
            ost,
            instance.setup,
            instance.mesh,
            instance.input_data,
        )

        options = _standardize_optimize_options(options)

        if algorithm == "sbs":

            _optimize_sbs(
                instance,
                control_vector,
                mapping,
                jobs_fun,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "Step By Step Optimization"

        elif algorithm == "l-bfgs-b":

            _optimize_lbfgsb(
                instance,
                control_vector,
                mapping,
                jobs_fun,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "L-BFGS-B Optimization"

        elif algorithm == "nelder-mead":

            _optimize_nelder_mead(
                instance,
                control_vector,
                mapping,
                jobs_fun,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "Nelder-Mead Optimization"

        #% TODO
        # elif algorithm == "nsga":

        if not inplace:

            return instance
