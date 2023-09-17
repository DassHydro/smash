from __future__ import annotations

from smash._constant import (
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    DEFAULT_BOUNDS_OPR_PARAMETERS,
    DEFAULT_BOUNDS_OPR_INITIAL_STATES,
)

from smash.core.model._build_model import (
    _map_dict_to_object,
    _build_setup,
    _build_mesh,
    _build_input_data,
    _build_parameters,
    _build_output,
)
from smash.core.model._standardize import (
    _standardize_get_opr_parameters_args,
    _standardize_get_opr_initial_states_args,
    _standardize_get_opr_final_states_args,
    _standardize_set_opr_parameters_args,
    _standardize_set_opr_initial_states_args,
)
from smash.core.simulation.run.run import _forward_run
from smash.core.simulation.run._standardize import _standardize_forward_run_args
from smash.core.simulation.optimize.optimize import _optimize
from smash.core.simulation.optimize._standardize import _standardize_optimize_args
from smash.core.simulation.estimate._standardize import (
    _standardize_multiset_estimate_args,
)
from smash.core.simulation.estimate.estimate import _multiset_estimate

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT

import numpy as np

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.simulation.optimize.optimize import MultipleOptimize, Optimize
    from smash.core.simulation.run.run import MultipleForwardRun, ForwardRun
    from smash.core.simulation.estimate.estimate import MultisetEstimate

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.

    Parameters
    ----------
    setup : dict
        Model initialization setup dictionary (TODO: add reference).

    mesh : dict
        Model initialization mesh dictionary (TODO: add reference).

    See Also
    --------
    smash.io.save_model : Save the Model object.
    smash.io.read_model : Read the Model object.
    smash.io.save_setup : Save the Model initialization setup dictionary.
    smash.io.read_setup : Read the Model initialization setup dictionary.
    smash.factory.generate_mesh : Automatic mesh generation.
    smash.io.save_mesh : Save the Model initialization mesh dictionary.
    smash.io.read_mesh : Read the Model initialization mesh dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    </> Reading precipitation: 100%|█████████| 1440/1440 [00:00<00:00, 10512.83it/s]
    </> Reading daily interannual pet: 100%|███| 366/366 [00:00<00:00, 13638.56it/s]
    </> Disaggregating daily interannual pet: 100%|█| 1440/1440 [00:00<00:00, 129442
    """

    def __init__(self, setup: dict | None, mesh: dict | None):
        if setup and mesh:
            if isinstance(setup, dict):
                nd = np.array(setup.get("descriptor_name", [])).size

                self.setup = SetupDT(nd)

                _map_dict_to_object(setup, self.setup)

                _build_setup(self.setup)

            else:
                raise TypeError(f"setup argument must be dict")

            if isinstance(mesh, dict):
                self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

                _map_dict_to_object(mesh, self.mesh)

                _build_mesh(self.setup, self.mesh)

            else:
                raise TypeError(f"mesh argument must be dict")

            self._input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self._input_data)

            self._parameters = ParametersDT(self.setup, self.mesh)

            _build_parameters(self.setup, self.mesh, self._input_data, self._parameters)

            self._output = OutputDT(self.setup, self.mesh)

            _build_output(self.setup, self._output)

    def __copy__(self):
        copy = Model(None, None)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy._input_data = self._input_data.copy()
        copy._parameters = self._parameters.copy()
        copy._output = self._output.copy()

        return copy

    @property
    def setup(self):
        """
        The setup used to create the Model object.

        Examples
        --------
        TODO: Fill
        """

        return self._setup

    @setup.setter
    def setup(self, value):
        self._setup = value

    @property
    def mesh(self):
        """
        The mesh used to create the Model object.

        Examples
        --------
        TODO: Fill
        """

        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def response_data(self):
        """
        Observation response data.

        Examples
        --------
        TODO: Fill
        """

        return self._input_data.response_data

    @response_data.setter
    def response_data(self, value):
        self._input_data.response_data = value

    @property
    def physio_data(self):
        """
        Physiographic data.

        Examples
        --------
        TODO: Fill
        """

        return self._input_data.physio_data

    @physio_data.setter
    def physio_data(self, value):
        self._input_data.physio_data = value

    @property
    def atmos_data(self):
        """
        Atmospheric and meteorological data.

        Examples
        --------
        TODO: Fill
        """

        return self._input_data.atmos_data

    @atmos_data.setter
    def atmos_data(self, value):
        self._input_data.atmos_data = value

    @property
    def opr_parameters(self):
        """
        Get operator parameters for the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return self._parameters.opr_parameters

    @opr_parameters.setter
    def opr_parameters(self, value):
        self._parameters.opr_parameters = value

    @property
    def opr_initial_states(self):
        """
        Get operator initial states for the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return self._parameters.opr_initial_states

    @opr_initial_states.setter
    def opr_initial_states(self, value):
        self._parameters.opr_initial_states = value

    @property
    def response(self):
        """
        Simulated response data.

        Examples
        --------
        TODO: Fill
        """

        return self._output.response

    @response.setter
    def response(self, value):
        self._output.response = value

    @property
    def opr_final_states(self):
        """
        Get operator final states for the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return self._output.opr_final_states

    @opr_final_states.setter
    def opr_final_states(self, value):
        self._output.opr_final_states = value

    def copy(self):
        """
        Make a deepcopy of the Model.

        Returns
        -------
        Model
            A copy of Model.

        Examples
        --------
        TODO: Fill
        """

        return self.__copy__()

    def get_opr_parameters(self, key: str) -> np.ndarray:
        """
        Get the values of an operator model parameter.

        Parameters
        ----------
        key : str
            The name of the operator parameter.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator parameter.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_parameters : Get operator parameters for the actual structure of the Model.
        """

        key = _standardize_get_opr_parameters_args(self, key)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        return self._parameters.opr_parameters.values[..., ind]

    def set_opr_parameters(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for an operator model parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the operator parameter.

        value : Numeric or np.ndarray
            The values to set for the operator parameter.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_parameters : Get operator parameters for the actual structure of the Model.
        """

        key, value = _standardize_set_opr_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        self._parameters.opr_parameters.values[..., ind] = value

    def get_opr_initial_states(self, key: str) -> np.ndarray:
        """
        Get the values of an operator model initial state.

        Parameters
        ----------
        key : str
            The name of the operator initial state.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator initial state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_initial_states : Get operator initial states for the actual structure of the Model.
        """

        key = _standardize_get_opr_initial_states_args(self, key)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        return self._parameters.opr_initial_states.values[..., ind]

    def set_opr_initial_states(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for an operator model initial state.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the operator initial state.

        value : Numeric or np.ndarray
            The values to set for the operator initial state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_initial_states : Get operator initial states for the actual structure of the Model.
        """

        key, value = _standardize_set_opr_initial_states_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        self._parameters.opr_initial_states.values[..., ind] = value

    def get_opr_final_states(self, key: str) -> np.ndarray:
        """
        Get the values of an operator model final state.

        Parameters
        ----------
        key : str
            The name of the operator final state.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator final state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_final_states : Get operator final states for the actual structure of the Model.
        """

        key = _standardize_get_opr_final_states_args(self, key)
        ind = np.argwhere(self._output.opr_final_states.keys == key).item()

        return self._output.opr_final_states.values[..., ind]

    def get_opr_parameters_bounds(self) -> dict:
        """
        Get the boundary condition for the operator model parameters.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each operator parameter in the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_PARAMETERS.items()
            if key in STRUCTURE_OPR_PARAMETERS[self.setup.structure]
        }

    def get_opr_initial_states_bounds(self) -> dict:
        """
        Get the boundary condition for the operator model initial states.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each operator initial state in the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_INITIAL_STATES.items()
            if key in STRUCTURE_OPR_STATES[self.setup.structure]
        }

    def forward_run(
        self,
        cost_options: dict | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> ForwardRun | None:
        """
        Run the forward Model.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        cost_options : dict or None, default None
            Dictionary containing computation cost options for simulated and observed responses. The elements are:

            jobs_cmpt : str or ListLike, default 'nse'
                Type of observation objective function(s) to be computed. Should be one or a sequence of any of

                - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
                - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
                - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

                .. hint::
                    See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

            wjobs_cmpt : str, Numeric, or ListLike, default 'mean'
                The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). The default is set to the average weighting.

            gauge : str or ListLike, default 'dws'
                Type of gauge to be computed. There are two ways to specify it:

                - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
                - An alias among 'all' (all gauge codes) and 'dws' (most downstream gauge code(s)).

            wgauge : str or ListLike, default 'mean'
                Type of gauge weights. There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of gauges optimized.
                - An alias among 'mean', 'lquartile' (1st quantile or lower quantile), 'median', or 'uquartile' (3rd quantile or upper quantile).

            event_seg : dict, default {'peak_quant': 0.995, 'max_duration': 240}
                A dictionary of event segmentation options when calculating flood event signatures for cost computation (i.e., **jobs_cmpt** includes flood events signatures).
                See `smash.hydrograph_segmentation` for more.

            end_warmup : str or pandas.Timestamp, default model.setup.start_time
                The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-23-05', '1998-23-06'])

                .. note::
                    It only applies to the following variables: 'opr_states' and 'q_domain'

            opr_states : bool, default False
                Whether to return operator states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_forward_run : ForwardRun or None, default None
            It returns a `smash.ForwardRun` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

        See Also
        --------
        smash.forward_run : Run the forward Model.
        ForwardRun : Represents forward run optional results.
        """

        args_options = [
            deepcopy(arg) for arg in [cost_options, common_options, return_options]
        ]

        args = _standardize_forward_run_args(self, *args_options)

        return _forward_run(self, *args)

    def optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict | None = None,
        cost_options: dict | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> Optimize | None:
        """
        Model assimilation using numerical optimization algorithms.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        mapping : str, default 'uniform'
            Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial', 'ann'.

        optimizer : str or None, default None
            Name of optimizer. Should be one of 'sbs', 'lbfgsb', 'sgd', 'adam', 'adagrad', 'rmsprop'.

            .. note::
                If not given, a default optimizer will be set depending on the optimization mapping:

                - **mapping** = 'uniform'; **optimizer** = 'sbs'
                - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'
                - **mapping** = 'ann'; **optimizer** = 'adam'

        optimize_options : dict or None, default None
            Dictionary containing optimization options for fine-tuning the optimization process.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See the returned parameters in `smash.default_optimize_options` for more.

        cost_options : dict or None, default None
            Dictionary containing computation cost options for simulated and observed responses. The elements are:

            jobs_cmpt : str or ListLike, default 'nse'
                Type of observation objective function(s) to be minimized. Should be one or a sequence of any of

                - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
                - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
                - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

                .. hint::
                    See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

            wjobs_cmpt : str, Numeric, or ListLike, default 'mean'
                The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). The default is set to the average weighting.

            wjreg : Numeric, default 0
                The weighting of regularization term. Only used with distributed mapping.

            jreg_cmpt : str or ListLike, default 'prior'
                Type(s) of regularization function(s) to be minimized when regularization term is set (i.e., **wjreg** > 0). Should be one or a sequence of any of 'prior' and 'smoothing'.

            wjreg_cmpt : str, Numeric, or ListLike, default 'mean'
                The corresponding weighting of regularization functions in case of multi-regularization (i.e., a sequence of regularization functions to compute). The default is set to the average weighting.

            gauge : str or ListLike, default 'dws'
                Type of gauge to be computed. There are two ways to specify it:

                - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
                - An alias among 'all' (all gauge codes) and 'dws' (most downstream gauge code(s)).

            wgauge : str or ListLike, default 'mean'
                Type of gauge weights. There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of gauges optimized.
                - An alias among 'mean', 'lquartile' (1st quantile or lower quantile), 'median', or 'uquartile' (3rd quantile or upper quantile).

            event_seg : dict, default {'peak_quant': 0.995, 'max_duration': 240}
                A dictionary of event segmentation options when calculating flood event signatures for cost computation (i.e., **jobs_cmpt** includes flood events signatures).
                See `smash.hydrograph_segmentation` for more.

            end_warmup : str or pandas.Timestamp, default model.setup.start_time
                The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-23-05', '1998-23-06'])

                .. note::
                    It only applies to the following variables: 'opr_states' and 'q_domain'

            opr_states : bool, default False
                Whether to return operator states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            iter_cost : bool, default False
                Whether to return cost iteration values.

            iter_projg : bool, default False
                Whether to return infinity norm of the projected gradient iteration values.

            control_vector : bool, default False
                Whether to return control vector at end of optimization. In case of optimization with ANN-based mapping, the control vector is represented in `smash.factory.Net.layers` instead.

            net : Net, default False
                Whether to return the trained neural network `smash.factory.Net`. Only used with ANN-based mapping.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            jreg : bool, default False
                Whether to return jreg (regularization component of cost) value.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_optimize : Optimize or None, default None
            It returns a `smash.Optimize` object containing the intermediate variables defined in **return_options**. If no intermediate variables
            are defined, it returns None.

        See Also
        --------
        smash.optimize : Model assimilation using numerical optimization algorithms.
        Optimize : Represents optimize optional results.
        """

        args_options = [
            deepcopy(arg)
            for arg in [optimize_options, cost_options, common_options, return_options]
        ]

        args = _standardize_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
        )

        return _optimize(self, *args)

    def multiset_estimate(
        self,
        multiset: MultipleForwardRun | MultipleOptimize,
        alpha: Numeric | ListLike | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> MultisetEstimate | None:
        """
        Model assimilation using Bayesian-like estimation on multiple sets of solutions.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        multiset : MultipleForwardRun or MultipleOptimize
            The returned object created by the `smash.multiple_forward_run` or `smash.multiple_optimize` method containing information about multiple sets of operator parameters or initial states.

        alpha : Numeric, ListLike, or None, default None
            A regularization parameter that controls the decay rate of the likelihood function. If **alpha** is a list-like object, the L-curve approach will be used to find an optimal value for the regularization parameter.

            .. note:: If not given, a default numeric range will be set for optimization through the L-curve process.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-23-05', '1998-23-06'])

                .. note::
                    It only applies to the following variables: 'opr_states' and 'q_domain'

            opr_states : bool, default False
                Whether to return operator states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            lcurve_multiset : bool, default False
                Whether to return the multiset estimate lcurve.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_multiset_estimate : MultisetEstimate or None, default None
            It returns a `smash.MultisetEstimate` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

        See Also
        --------
        smash.multiset_estimate : Model assimilation using Bayesian-like estimation on multiple sets of solutions.
        MultisetEstimate : Represents multiset estimate optional results.
        MultipleForwardRun : Represents multiple forward run computation result.
        MultipleOptimize : Represents multiple optimize computation result.
        """

        args_options = [deepcopy(arg) for arg in [common_options, return_options]]

        args = _standardize_multiset_estimate_args(self, multiset, alpha, *args_options)

        return _multiset_estimate(self, *args)
