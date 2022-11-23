from __future__ import annotations

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mw_forward import forward

import os
import errno

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

from smash.core._event_segmentation import _segmentation

from smash.core._signatures import (
    _standardize_signatures,
    _signatures_one_basin,
    _signatures_sensitivity_one_basin,
)

from smash.core.generate_samples import generate_samples, _model2problem

from smash.io.model_io import save_model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.

    Parameters
    ----------
    setup : dict
        Model initialization setup dictionary (see: :ref:`setup arguments <user_guide.model_initialization.setup>`).

    mesh : dict
        Model initialization mesh dictionary. (see: :ref:`mesh arguments <user_guide.model_initialization.mesh>`).

    See Also
    --------
    save_setup: Save Model initialization setup dictionary.
    read_setup: Read Model initialization setup dictionary.
    save_mesh: Save Model initialization mesh dictionary.
    read_mesh: Read Model initialization mesh dictionary.
    generate_mesh: Automatic mesh generation.
    save_model: Save Model object.
    read_model: Read Model object.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization
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

    def event_segmentation(self):
        """
        Return a DataFrame containing segmentation information of flood events over all catchments of Model object.

        Returns
        -------
        df : pandas.DataFrame
            flood events information obtained from segmentation algorithm.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Perform segmentation algorithm and display flood events infomation:

        >>> df = model.event_segmentation()
        >>> df
          catchment               start                 end         maxrainfall               flood  season
        0  V3524010 2014-11-03 03:00:00 2014-11-10 15:00:00 2014-11-04 12:00:00 2014-11-04 19:00:00  autumn
        1  V3515010 2014-11-03 10:00:00 2014-11-08 10:00:00 2014-11-04 12:00:00 2014-11-04 20:00:00  autumn
        2  V3517010 2014-11-03 08:00:00 2014-11-11 00:00:00 2014-11-04 11:00:00 2014-11-04 16:00:00  autumn
        """

        df = _segmentation(self.copy())

        return df

    def signatures(self, sign=None):

        """
        Compute continuous or/and flood event signatures of the Model.

        Parameters
        ----------
        sign : str, list or None, default None
                Define signature(s) to compute.
                List of all continuous and flood event signatures:

                    'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90',
                    'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf'

                If None, all of continuous and flood event signatures will be computed.

        Returns
        -------
        df : dict
                `df` with two DataFrames of i. observed and simulated continuous signatures and ii. observed and simulated flood event signatures.

        Sname : dict
                `Sname` with two lists of i. computed continuous signatures and ii. computed flood event signatures.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> model.optimize(inplace=True)

        Compute all continuous and flood event signatures:

        >>> df, Sname = model.signatures()
        >>> df["C"]
          catchment   Crc_obs  Crchf_obs  Crclf_obs  Crch2r_obs  Cfp2_obs  ...  Crclf_sim  Crch2r_sim      Cfp2_sim     Cfp10_sim  Cfp50_sim  Cfp90_sim
        0  V3524010  0.516207   0.191349   0.324854    0.370684   1.02456  ...   0.289188    0.411256  1.714191e-10  2.570761e-03   3.616916  39.241742
        1  V3515010  0.509180   0.147217   0.361951    0.289125   0.31100  ...   0.278125    0.385084  1.559969e-15  5.540762e-05   0.984099   9.691529
        2  V3517010  0.514302   0.148364   0.365926    0.288476   0.06500  ...   0.273115    0.398862  8.358947e-38  4.860812e-14   0.319221   2.687196

        [3 rows x 17 columns]

        >>> df["E"]
          catchment  season               start                 end    Eff_obs    Ebf_obs  ...   Erc_sim  Erchf_sim  Erclf_sim  Erch2r_sim Elt_sim     Epf_sim
        0  V3524010  autumn 2014-11-03 03:00:00 2014-11-10 15:00:00  28.146225  34.729198  ...  0.588040   0.288424   0.299594    0.490484       8  280.677338
        1  V3515010  autumn 2014-11-03 10:00:00 2014-11-08 10:00:00   7.068919   9.895721  ...  0.563045   0.279523   0.283470    0.496448       6   61.226574
        2  V3517010  autumn 2014-11-03 08:00:00 2014-11-11 00:00:00   1.336155   2.453123  ...  0.586809   0.275075   0.311709    0.468764       6   18.758123

        [3 rows x 20 columns]

        >>> Sname["C"]
        ['Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90']

        >>> Sname["E"]
        ['Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf']

        """

        instance = self.copy()

        CS, ES = _standardize_signatures(sign)

        df, Sname = _signatures_one_basin(instance, CS, ES)

        return df, Sname

    def signatures_sensitivity(
        self,
        N=64,
        sign=None,
        save=False,
        path=None,
    ):
        """
        Compute variance-based sensitivity (Sobol indices) of signatures.

        Parameters
        ----------
        N : int, default 64
                Number of trajectories to generate for each model parameter (ideally a power of 2).
                Then the number of sample to generate for all model parameters is equal to :math:`N(2D+2)`
                where :math:`D` is the number of model parameters.

                See `here <https://salib.readthedocs.io/en/latest/api.html>`__ for more details.

        sign : str, list or None, default None
                Define signature(s) to compute.
                List of all continuous and flood event signatures:

                    'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90',
                    'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf'

                If None, all of continuous and flood event signatures will be computed.

        save : bool, default False
                If True, save simulation results under HDF5 file.

        path : None, str, path object, or file-like object, default None
                `path` should be provided when ``save=True`` and correspond to a folder where the files will be saved.

        Returns
        -------
        sample : DataFrame
                Generated sample for sensititvity computation.

        dfSens : dict
                `dfSens` with two DataFrames of i. continuous signatures sensitivity and ii. flood event signatures sensitivity.

        Sname : dict
                `Sname` with two lists of i. computed continuous signatures and ii. computed flood event signatures.

        See Also
        --------
        save_model : Save Model object.

        Examples
        --------
        >>> setup, mesh = smash.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> res = model.signatures_sensitivity()

        Continuous signatures sensitivity computation:

        >>> res[1]["C"]
          catchment  Crc_sim.ST_cp  Crc_sim.ST_cft  Crc_sim.ST_exc  ...  Cfp90_sim.S2_cp-lr  Cfp90_sim.S2_cft-exc  Cfp90_sim.S2_cft-lr  Cfp90_sim.S2_exc-lr
        0  V3524010       0.025848        0.322964        0.492531  ...           -0.008993              0.604393             0.084209             0.081659
        1  V3515010       0.009877        0.288598        0.492263  ...           -0.011919              0.628929             0.111150             0.109329
        2  V3517010       0.009662        0.300603        0.518927  ...           -0.014917              0.609301             0.105303             0.120858

        [3 rows x 113 columns]

        Flood event signatures sensitivity computation:

        >>> res[1]["E"]
          catchment  season               start                 end  ...  Epf_sim.S2_cp-lr  Epf_sim.S2_cft-exc  Epf_sim.S2_cft-lr  Epf_sim.S2_exc-lr
        0  V3524010  autumn 2014-11-03 03:00:00 2014-11-10 15:00:00  ...         -0.007437            0.588443           0.056097           0.072928
        1  V3515010  autumn 2014-11-03 10:00:00 2014-11-08 10:00:00  ...         -0.004476            0.565173           0.073368           0.098639
        2  V3517010  autumn 2014-11-03 08:00:00 2014-11-11 00:00:00  ...         -0.009714            0.467997           0.041326           0.086835

        [3 rows x 116 columns]

        Lists of studied signatures:

        >>> print(res[2])
        {'C': ['Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90'], 'E': ['Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf']}
        
        """

        instance = self.copy()

        CS, ES = _standardize_signatures(sign)

        problem = _model2problem(instance.setup)
        sample = generate_samples(problem=problem, generator='saltelli', n=N)

        if save and not os.path.isdir(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        dfC = []
        dfE = []

        for i in range(len(sample)):

            for k in sample.keys():
                setattr(instance.parameters, k, sample[k][i])

            instance.run(inplace=True)

            if save:
                save_model(instance, f"{path}/run_{i+1}.hdf5")

            df, Sname = _signatures_one_basin(instance, CS, ES, sign_obs=False)

            dfC += [df["C"]]
            dfE += [df["E"]]

        dfSens = _signatures_sensitivity_one_basin(problem, {"C": dfC, "E": dfE}, Sname)

        return sample, dfSens, Sname

