from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from smash._constant import (
    DEFAULT_BOUNDS_RR_INITIAL_STATES,
    DEFAULT_BOUNDS_RR_PARAMETERS,
    DEFAULT_BOUNDS_SERR_MU_PARAMETERS,
    DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
)
from smash.core.model._build_model import (
    _adjust_interception,
    _build_input_data,
    _build_mesh,
    _build_output,
    _build_parameters,
    _map_dict_to_fortran_derived_type,
)
from smash.core.model._standardize import (
    _standardize_get_rr_final_states_args,
    _standardize_get_rr_initial_states_args,
    _standardize_get_rr_parameters_args,
    _standardize_get_serr_mu_parameters_args,
    _standardize_get_serr_sigma_parameters_args,
    _standardize_model_args,
    _standardize_set_nn_parameters_bias_args,
    _standardize_set_nn_parameters_weight_args,
    _standardize_set_rr_initial_states_args,
    _standardize_set_rr_parameters_args,
    _standardize_set_serr_mu_parameters_args,
    _standardize_set_serr_sigma_parameters_args,
)
from smash.core.simulation._doc import (
    _bayesian_optimize_doc_appender,
    _forward_run_doc_appender,
    _model_bayesian_optimize_doc_substitution,
    _model_forward_run_doc_substitution,
    _model_multiset_estimate_doc_substitution,
    _model_optimize_doc_substitution,
    _multiset_estimate_doc_appender,
    _optimize_doc_appender,
    _set_control_bayesian_optimize_doc_appender,
    _set_control_bayesian_optimize_doc_substitution,
    _set_control_optimize_doc_appender,
    _set_control_optimize_doc_substitution,
)
from smash.core.simulation.estimate._standardize import (
    _standardize_multiset_estimate_args,
)
from smash.core.simulation.estimate.estimate import _multiset_estimate
from smash.core.simulation.optimize._standardize import (
    _standardize_bayesian_optimize_args,
    _standardize_optimize_args,
)
from smash.core.simulation.optimize._tools import _set_control
from smash.core.simulation.optimize.optimize import (
    _bayesian_optimize,
    _optimize,
)
from smash.core.simulation.run._standardize import _standardize_forward_run_args
from smash.core.simulation.run.run import _forward_run
from smash.factory.net._layers import _initialize_nn_parameter
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_output import OutputDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_parameters_manipulation import (
    get_serr_mu as wrap_get_serr_mu,
)
from smash.fcore._mwd_parameters_manipulation import (
    get_serr_sigma as wrap_get_serr_sigma,
)
from smash.fcore._mwd_setup import SetupDT

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from smash.core.simulation.estimate.estimate import MultisetEstimate
    from smash.core.simulation.optimize.optimize import (
        BayesianOptimize,
        Optimize,
    )
    from smash.core.simulation.run.run import ForwardRun, MultipleForwardRun
    from smash.fcore._mwd_atmos_data import Atmos_DataDT
    from smash.fcore._mwd_nn_parameters import NN_ParametersDT
    from smash.fcore._mwd_physio_data import Physio_DataDT
    from smash.fcore._mwd_response import ResponseDT
    from smash.fcore._mwd_response_data import Response_DataDT
    from smash.fcore._mwd_rr_parameters import RR_ParametersDT
    from smash.fcore._mwd_rr_states import RR_StatesDT
    from smash.fcore._mwd_serr_mu_parameters import SErr_Mu_ParametersDT
    from smash.fcore._mwd_serr_sigma_parameters import SErr_Sigma_ParametersDT
    from smash.fcore._mwd_u_response_data import U_Response_DataDT
    from smash.util._typing import ListLike, Numeric

__all__ = ["Model"]


class Model:
    """
    Primary data structure of the hydrological model `smash`.

    Parameters
    ----------
    setup : `dict[str, Any]`
        Model initialization setup dictionary. The elements are:

        snow_module : `str`, default 'zero'
            Name of snow module. Should be one of:

            - ``'zero'``
            - ``'ssn'``

            .. hint::
                See the :ref:`Snow Module <math_num_documentation.forward_structure.snow_module>` section

        hydrological_module : `str`, default 'gr4'
            Name of hydrological module. Should be one of:

            - ``'gr4'``, ``'gr4_mlp'``, ``'gr4_ri'``, ``'gr4_ode'``, ``'gr4_ode_mlp'``
            - ``'gr5'``, ``'gr5_mlp'``, ``'gr5_ri'``
            - ``'gr6'``, ``'gr6_mlp'``
            - ``'grc'``, ``'grc_mlp'``
            - ``'grd'``, ``'grd_mlp'``
            - ``'loieau'``, ``'loieau_mlp'``
            - ``'vic3l'``

            .. hint::
                See the :ref:`Hydrological Module
                <math_num_documentation.forward_structure.hydrological_module>` section

        routing_module : `str`, default 'lr'
            Name of routing module. Should be one of:

            - ``'lag0'``
            - ``'lr'``
            - ``'kw'``

            .. hint::
                See the :ref:`Routing Module <math_num_documentation.forward_structure.routing_module>`
                section

        hidden_neuron : `int` or `list[int]`, default 16
            Number of neurons in hidden layer(s) of the parameterization neural network
            used to correct internal fluxes, if used (depending on **hydrological_module**).
            If it is a list, the maximum length is 2,
            which means the neural network can have up to 2 hidden layers.

        serr_mu_mapping : `str`, default 'Zero'
            Name of the mapping used for :math:`\\mu`, the mean of structural errors. Should be one of:

            - ``'Zero'`` (:math:`\\mu = 0`)
            - ``'Constant'`` (:math:`\\mu = \\mu_0`)
            - ``'Linear'`` (:math:`\\mu = \\mu_0 + \\mu_1 \\times q`)

            .. hint::
                See the :ref:`math_num_documentation.bayesian_estimation` section

        serr_sigma_mapping : `str`, default 'Linear'
            Name of the mapping used for :math:`\\sigma`, the standard deviation of structural errors.
            Should be one of:

            - ``'Constant'`` (:math:`\\sigma=\\sigma_0`)
            - ``'Linear'`` (:math:`\\sigma=\\sigma_0 + \\sigma_1 \\times q`)
            - ``'Power'`` (:math:`\\sigma=\\sigma_0 + \\sigma_1 \\times q^{\\sigma_2}`)
            - ``'Exponential'`` (:math:`\\sigma=\\sigma_0 + (\\sigma_2-\\sigma_0) \\times \\left( 1-\\exp
              (-q/\\sigma_1) \\right)`)
            - ``'Gaussian'`` (:math:`\\sigma=\\sigma_0 + (\\sigma_2-\\sigma_0) \\times \\left( 1-\\exp(
              -(q/\\sigma_1)^2) \\right)`)

            .. hint::
                See the :ref:`math_num_documentation.bayesian_estimation` section

        dt : `float`, default 3600
            Simulation time step in seconds.

        start_time : `str`, `datetime.date` or `pandas.Timestamp`
            Start time date.

        end_time : `str`, `datetime.date` or `pandas.Timestamp`
            End time date. **end_time** must be later than **start_time**

        .. note::
            The convention of `smash` is that **start_time** is the date used to initialize the model's
            states. All the modeled state-flux variables :math:`\\boldsymbol{U}(x,t)` (i.e., discharge,
            states, internal fluxes) will be computed over the period **start_time + 1dt** and **end_time**

        adjust_interception : `bool`, default True
            Whether or not to adjust the maximum capacity of the interception reservoir.
            This option is available for any **hydrological_module** having the :math:`c_i` parameter
            (i.e., ``'gr4'``, ``'gr5'``, ``'gr6'``, etc.) and for a sub-daily simulation time step **dt**.

        compute_mean_atmos : `bool`, default True
            Whether or not to compute mean atmospheric data for each gauge.

        read_qobs : `bool`, default False
            Whether or not to read observed discharge file(s).

            .. hint::
                See the :ref:`user_guide.data_and_format_description.format_description` section

        qobs_directory : `str`
            Path to the root directory of the observed discharge file(s).
            This option is ``mandatory`` if **read_qobs** is set to True.

        read_prcp : `bool`, default False
            Whether or not to read precipitation file(s).

            .. hint::
                See the :ref:`user_guide.data_and_format_description.format_description` section

        prcp_format : `str`, default 'tif'
            Precipitation file format. This option is only applicable if **read_prcp** is set to True.

            .. note::
                Only the ``tif`` format is currently supported. We would like to extend this to ``netcdf``
                format in future version.

        prcp_conversion_factor : `float`, default 1
            Preciptation conversion factor. The precipitation will be ``multiplied`` by the conversion factor.
            This option is only applicable if **read_prcp** is set to True.

        prcp_directory : `str`
            Path to the root directory of the precipitation file(s).
            This option is ``mandatory`` if **read_prcp** is set to True.

        prcp_access : `str`, default ''
            Precipitation directory structure access.
            By default, files are read using a recursive search from the root directory **prcp_directory**.
            This option makes it possible to specify the directory structure and allow faster access according
            to **start_time** and **end_time** dates.
            This option is only applicable if **read_prcp** is set to True.

        read_pet : `bool`, default False
            Whether or not to read potential evapotranspiration file(s).

            .. hint::
                See the :ref:`user_guide.data_and_format_description.format_description` section

        pet_format : `str`, default 'tif'
            Potential evapotranspiration file format. This option is only applicable if **read_pet** is set
            to True.

            .. note::
                Only the ``tif`` format is currently supported. We would like to extend this to ``netcdf``
                format in future version.

        pet_conversion_factor : `float`, default 1
            Potential evapotranspiration conversion factor. The potential evapotranspiration will be
            ``multiplied`` by the conversion factor.
            This option is only applicable if **read_pet** is set to True.

        pet_directory : `str`
            Path to the root directory of the potential evapotranspiration file(s).
            This option is ``mandatory`` if **read_pet** is set to True.

        pet_access : `str`, default ''
            Potential evapotranspiration directory structure access.
            By default, files are read using a recursive search from the root directory **pet_directory**.
            This option makes it possible to specify the directory structure and allow faster access according
            to **start_time** and **end_time** dates.
            This option is only applicable if **read_pet** is set to True.

        daily_interannual_pet : `bool`, default False
            Whether or not to read daily interannual potential evapotranspiration.
            This replaces the conventional way of reading a file in time steps.

        read_snow : `bool`, default False
            Whether or not to read snow file(s).
            This option is only applicable if **snow_module** is set to ``ssn``.

            .. hint::
                See the :ref:`user_guide.data_and_format_description.format_description` section

        snow_format : `str`, default 'tif'
            Snow file format. This option is only applicable if **read_snow** is set to True and if
            **snow_module** is set to ``ssn``.

            .. note::
                Only the ``tif`` format is currently supported. We would like to extend this to ``netcdf``
                format in future version.

        snow_conversion_factor : `float`, default 1
            Snow conversion factor. The snow will be ``multiplied`` by the conversion factor.
            This option is only applicable if **read_snow** is set to True and if **snow_module** is set to
            ``ssn``.

        snow_directory : `str`
            Path to the root directory of the snow file(s).
            This option is ``mandatory`` if **read_snow** is set to True and if **snow_module** is set to
            ``ssn``.

        snow_access : `str`, default ''
            Snow directory structure access.
            By default, files are read using a recursive search from the root directory **snow_directory**.
            This option makes it possible to specify the directory structure and allow faster access according
            to **start_time** and **end_time** dates.
            This option is only applicable if **read_snow** is set to True and if **snow_module** is set to
            ``ssn``.

        read_temp : `bool`, default False
            Whether or not to read temperature file(s).

            .. hint::
                See the :ref:`user_guide.data_and_format_description.format_description` section

        temp_format : `str`, default 'tif'
            Temperature file format. This option is only applicable if **read_temp** is set to True and if
            **snow_module** is set to ``ssn``.

            .. note::
                Only the ``tif`` format is currently supported. We would like to extend this to ``netcdf``
                format in future version.

        temp_directory : `str`
            Path to the root directory of the temperature file(s).
            This option is ``mandatory`` if **read_temp** is set to True and if **snow_module** is set to
            ``ssn``.

        temp_access : `str`, default ''
            Temperature directory structure access.
            By default, files are read using a recursive search from the root directory **temp_directory**.
            This option makes it possible to specify the directory structure and allow faster access according
            to **start_time** and **end_time** dates.
            This option is only applicable if **read_temp** is set to True and if **snow_module** is set to
            ``ssn``.

        prcp_partitioning : `bool`, default False
            Whether or not to partition precipitation into liquid (precipitation) and solid (snow) parts.
            If precipitation and snow are read, the precipitation and snow will be summed before partitioning.
            This option is only applicable if **snow_module** is set to ``ssn``.

            .. hint::
                See the :ref:`math_num_documentation.precipitation_partitioning` section

        sparse_storage : `bool`, default False
            Whether or not to store atmospheric data (i.e., precipitation, potential evapotranspiration, snow
            and temperature) sparsely.
            This option reduces the amount of memory taken up by atmospheric data. It is particularly useful
            when working large dataset.

        read_descriptor : `bool`, default False
            Whether or not to read descriptor file(s).

        descriptor_format : `str`, default 'tif'
            Descriptor file format. This option is only applicable if **read_descriptor** is set to True.

            .. note::
                Only the ``tif`` format is currently supported. We would like to extend this to ``netcdf``
                format in future version.

        descriptor_directory : `str`
            Path to the root directory of the descriptor file(s).
            This option is ``mandatory`` if **read_descriptor** is set to True.

        descriptor_name : `list[str]`
            List of descriptor name.
            This option is ``mandatory`` if **read_descriptor** is set to True.

        read_imperviousness : `bool`, default False
            Whether or not to read descriptor file(s).

        imperviousness_format : `str`, default 'tif'
            This option is only applicable if **read_imperviousness** is set to True.

            .. note::
                Only the ``tif`` format is currently supported.

        imperviousness_directory : `str`
            Path to the imperviousness file.
            This option is ``mandatory`` if **read_imperviousness** is set to True.

    mesh : `dict[str, Any]`
        Model initialization mesh dictionary.

        .. note::
            The elements are described in the `smash.factory.generate_mesh <factory.generate_mesh>` method.

    See Also
    --------
    smash.io.save_setup : Save the Model initialization setup dictionary to YAML.
    smash.io.read_setup : Read the Model initialization setup dictionary from YAML.
    smash.factory.generate_mesh : Automatic Model initialization mesh dictionary generation.
    smash.io.save_mesh : Save the Model initialization mesh dictionary to HDF5.
    smash.io.read_mesh : Read the Model initialization mesh dictionary from HDF5.
    smash.io.save_model : Save the Model object to HDF5.
    smash.io.read_model : Read the Model object from HDF5.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")

    Setup and mesh dictionaries loaded from ``Cance`` dataset

    >>> setup
    {
        'hydrological_module': 'gr4',
        'routing_module': 'lr',
        'dt': 3600,
        ...
        'descriptor_name': ['slope', 'dd']
    }
    >>> mesh.keys()
    dict_keys(['active_cell', 'area', 'area_dln', ..., 'yres'])

    Constructing the Model object

    >>> model = smash.Model(setup, mesh)
    </> Reading precipitation: 100%|█████████████████████████████| 1440/1440 [00:00<00:00, 10323.28it/s]
    </> Reading daily interannual pet: 100%|███████████████████████| 366/366 [00:00<00:00, 13735.82it/s]
    </> Disaggregating daily interannual pet: 100%|█████████████| 1440/1440 [00:00<00:00, 132565.08it/s]
    </> Computing mean atmospheric data
    </> Adjusting GR interception capacity
    >>> model
    Model
        atmos_data: ['mean_pet', 'mean_prcp', '...', 'sparse_prcp', 'sparse_snow']
        mesh: ['active_cell', 'area', '...', 'xres', 'ymax']
        physio_data: ['descriptor', 'l_descriptor', 'u_descriptor']
        response: ['q']
        response_data: ['q']
        rr_final_states: ['keys', 'values']
        rr_initial_states: ['keys', 'values']
        rr_parameters: ['keys', 'values']
        serr_mu_parameters: ['keys', 'values']
        serr_sigma_parameters: ['keys', 'values']
        setup: ['adjust_interception', 'compute_mean_atmos', '...', 'temp_access', 'temp_directory']
        u_response_data: ['q_stdev']
    """

    def __init__(self, setup: dict[str, Any] | None, mesh: dict[str, Any] | None):
        if setup and mesh:
            args = [deepcopy(arg) for arg in [setup, mesh]]
            setup, mesh = _standardize_model_args(*args)

            self.setup = SetupDT(setup["nd"])

            _map_dict_to_fortran_derived_type(setup, self.setup)

            self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["npar"], mesh["ng"])

            _map_dict_to_fortran_derived_type(mesh, self.mesh)

            _build_mesh(self.setup, self.mesh)

            self._input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self._input_data)

            self._parameters = ParametersDT(self.setup, self.mesh)

            _build_parameters(self.setup, self.mesh, self._input_data, self._parameters)

            self._output = OutputDT(self.setup, self.mesh)

            _build_output(self.setup, self._output)

    def __copy__(self) -> Model:
        copy = Model(None, None)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy._input_data = self._input_data.copy()
        copy._parameters = self._parameters.copy()
        copy._output = self._output.copy()

        return copy

    def __repr__(self) -> str:
        # % Nested function. This avoids duplicating the attribute check.
        def _valid_attr(obj, attr):
            if attr.startswith("_"):
                return False
            try:
                value = getattr(obj, attr)
            except Exception:
                return False
            if callable(value):
                return False

            return True

        ret = [self.__class__.__name__]
        for attr in dir(self):
            if not _valid_attr(self, attr):
                continue
            value = getattr(self, attr)

            sub_attr_list = [sub_attr for sub_attr in dir(value) if _valid_attr(value, sub_attr)]

            # % Do not print too much attributes
            if len(sub_attr_list) > 4:
                sub_attr_list = sub_attr_list[0:2] + ["..."] + sub_attr_list[-3:-1]

            ret.append(f"    {attr}: {sub_attr_list}")

        return "\n".join(ret)

    @property
    def setup(self) -> SetupDT:
        """
        Model setup.

        Returns
        -------
        setup : `SetupDT <fcore._mwd_setup.SetupDT>`
            It returns a Fortran derived type containing the variables relating to the setup.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model setup

        >>> model.setup
        SetupDT
            adjust_interception: 1
            compute_mean_atmos: 1
            ...
            temp_directory: '...'
            temp_format: 'tif'

        Access to specific values

        >>> model.setup.dt, model.setup.hydrological_module
        (3600.0, 'gr4')

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.setup.<TAB>
        model.setup.adjust_interception     model.setup.pet_format
        model.setup.compute_mean_atmos      model.setup.prcp_access
        model.setup.copy()                  model.setup.prcp_conversion_factor
        model.setup.daily_interannual_pet   model.setup.prcp_directory
        model.setup.descriptor_directory    model.setup.prcp_format
        model.setup.descriptor_format       model.setup.prcp_partitioning
        model.setup.descriptor_name         model.setup.qobs_directory
        model.setup.dt                      model.setup.read_descriptor
        model.setup.end_time                model.setup.read_pet
        model.setup.from_handle(            model.setup.read_prcp
        model.setup.hidden_neuron           model.setup.read_qobs
        model.setup.hydrological_module     model.setup.read_snow
        model.setup.n_hydro_fluxes          model.setup.read_temp
        model.setup.n_internal_fluxes       model.setup.routing_module
        model.setup.n_layers                model.setup.serr_mu_mapping
        model.setup.n_routing_fluxes        model.setup.serr_sigma_mapping
        model.setup.n_snow_fluxes           model.setup.snow_access
        model.setup.nd                      model.setup.snow_conversion_factor
        model.setup.neurons                 model.setup.snow_directory
        model.setup.nqz                     model.setup.snow_format
        model.setup.nrrp                    model.setup.snow_module
        model.setup.nrrs                    model.setup.snow_module_present
        model.setup.nsep_mu                 model.setup.sparse_storage
        model.setup.nsep_sigma              model.setup.start_time
        model.setup.ntime_step              model.setup.structure
        model.setup.pet_access              model.setup.temp_access
        model.setup.pet_conversion_factor   model.setup.temp_directory
        model.setup.pet_directory           model.setup.temp_format
        """

        return self._setup

    @setup.setter
    def setup(self, value: SetupDT):
        self._setup = value

    @property
    def mesh(self) -> MeshDT:
        """
        Model mesh.

        Returns
        -------
        mesh : `MeshDT <fcore._mwd_mesh.MeshDT>`
            It returns a Fortran derived type containing the variables relating to the mesh.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model mesh

        >>> model.mesh
        MeshDT
            active_cell: array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0],
            ...
            ymax: 6478000.0
            yres: 1000.0

        Access to specific values

        >>> model.mesh.nrow, model.setup.ncol
        (28, 28)
        >>> model.mesh.flwdir
        array([[1, 5, 6, 1, 1, 8, 8, 1, 1, 8, 8, 1, 5, 6, 5, 3, 2, 2, 3, 3, 3, 3,
        3, 4, 5, 8, 5, 6],
        ...
        [2, 1, 1, 8, 5, 5, 6, 2, 1, 1, 4, 3, 2, 2, 7, 1, 2, 2, 2, 1, 4, 3,
        2, 1, 3, 2, 2, 5]], dtype=int32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.mesh.<TAB>
        model.mesh.active_cell           model.mesh.gauge_pos
        model.mesh.area                  model.mesh.local_active_cell
        model.mesh.area_dln              model.mesh.nac
        model.mesh.code                  model.mesh.ncol
        model.mesh.copy()                model.mesh.ncpar
        model.mesh.cpar_to_rowcol        model.mesh.ng
        model.mesh.cscpar                model.mesh.npar
        model.mesh.dx                    model.mesh.nrow
        model.mesh.dy                    model.mesh.rowcol_to_ind_ac
        model.mesh.flwacc                model.mesh.xmin
        model.mesh.flwdir                model.mesh.xres
        model.mesh.flwdst                model.mesh.ymax
        model.mesh.flwpar                model.mesh.yres
        model.mesh.from_handle(
        """

        return self._mesh

    @mesh.setter
    def mesh(self, value: MeshDT):
        self._mesh = value

    @property
    def response_data(self) -> Response_DataDT:
        """
        Model response data.

        Returns
        -------
        response_data : `Response_DataDT <fcore._mwd_response_data.Response_DataDT>`
            It returns a Fortran derived type containing the variables relating to the response data.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model response data

        >>> model.response_data
        Response_DataDT
            q: array([[ 1.237,  1.232,  1.224, ..., 22.951, 22.813, 22.691],
            [ 0.38 ,  0.382,  0.385, ...,  6.789,  6.759,  6.729],
            [ 0.094,  0.094,  0.094, ...,  1.588,  1.578,  1.568]],
            dtype=float32)

        Access to a specific gauge observed discharge time serie

        >>> model.mesh.code
        array(['V3524010', 'V3515010', 'V3517010'], dtype='<U8')
        >>> ind = np.argwhere(model.mesh.code == "V3524010").item()
        >>> ind
        0
        >>> model.response_data.q[ind, :]
        array([ 1.237,  1.232,  1.224, ..., 22.951, 22.813, 22.691], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.response_data.<TAB>
        model.response_data.copy()        model.response_data.q
        model.response_data.from_handle(
        """

        return self._input_data.response_data

    @response_data.setter
    def response_data(self, value: Response_DataDT):
        self._input_data.response_data = value

    @property
    def u_response_data(self) -> U_Response_DataDT:
        """
        Model response data uncertainties.

        Returns
        -------
        u_response_data : `U_Response_DataDT <fcore._mwd_u_response_data.U_Response_DataDT>`
            It returns a Fortran derived type containing the variables relating to the response data
            uncertainties.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model response data uncertainties

        >>> model.u_response_data
        U_Response_DataDT
            q_stdev: array([[0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

        Access to a specific gauge discharge uncertainties (standard deviation of independent error) time
        serie

        >>> model.mesh.code
        array(['V3524010', 'V3515010', 'V3517010'], dtype='<U8')
        >>> ind = np.argwhere(model.mesh.code == "V3524010").item()
        >>> ind
        0
        >>> model.u_response_data.q_stdev[ind, :]
        array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)

        Set discharge uncertainties proportional to observation discharge

        >>> model.u_response_data.q_stdev = model.response_data.q * 0.1
        >>> model.u_response_data.q_stdev
        array([[0.1237, 0.1232, 0.1224, ..., 2.2951, 2.2813, 2.2691],
               [0.038 , 0.0382, 0.0385, ..., 0.6789, 0.6759, 0.6729],
               [0.0094, 0.0094, 0.0094, ..., 0.1588, 0.1578, 0.1568]],
              dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.response_data.<TAB>
        model.response_data.copy()        model.response_data.q_stdev
        model.response_data.from_handle(
        """

        return self._input_data.u_response_data

    @u_response_data.setter
    def u_response_data(self, value: U_Response_DataDT):
        self._input_data.u_response_data = value

    @property
    def physio_data(self) -> Physio_DataDT:
        """
        Model physiographic data.

        Returns
        -------
        physio_data : `Physio_DataDT <fcore._mwd_physio_data.Physio_DataDT>`
            It returns a Fortran derived type containing the variables relating to the physiographic data.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model physiograhic data

        >>> model.physio_data
        Physio_DataDT
            descriptor: array([[[1.6998979e+00, 1.5327297e+01],
                [9.5237291e-01, 1.4062435e+01],
                ...
                [1.2933834e+00, 2.0580339e+01],
                [1.3551705e+00, 2.1825863e+01]]], dtype=float32)
            l_descriptor: array([0.       , 3.0111098], dtype=float32)
            u_descriptor: array([ 5.455888, 23.433908], dtype=float32)

        Access to a specific physiographic descriptor

        >>> model.setup.descriptor_name
        array(['slope', 'dd'], dtype='<U5')
        >>> ind = np.argwhere(model.setup.descriptor_name == "slope").item()
        >>> ind
        0
        >>> model.physio_data.descriptor[..., ind]
        array([[1.69989789e+00, 9.52372909e-01, 2.01547050e+00, 3.75710177e+00,
                3.41233420e+00, 2.92353439e+00, 2.85195327e+00, 2.82403517e+00,
                ...
                2.60070384e-01, 4.05688077e-01, 1.29338336e+00, 1.35517049e+00]],
              dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.physio_data.<TAB>
        model.physio_data.copy()        model.physio_data.l_descriptor
        model.physio_data.descriptor    model.physio_data.u_descriptor
        model.physio_data.from_handle(
        """

        return self._input_data.physio_data

    @physio_data.setter
    def physio_data(self, value: Physio_DataDT):
        self._input_data.physio_data = value

    @property
    def atmos_data(self) -> Atmos_DataDT:
        """
        Model atmospheric data.

        Returns
        -------
        atmos_data : `Atmos_DataDT <fcore._mwd_atmos_data.Atmos_DataDT>`
            It returns a Fortran derived type containing the variables relating to the atmospheric data.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model atmospheric data

        >>> model.atmos_data
        Atmos_DataDT
            mean_pet: array([[0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
            mean_prcp: array([[0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
            pet: array([[[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...
                [0., 0., 0., ..., 0., 0., 0.]]], dtype=float32)
            prcp: array([[[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...
                [0., 0., 0., ..., 0., 0., 0.]]], dtype=float32)

        .. warning::
            If the Model object has been initialised with the ``sparse_storage`` option in setup
            (see `Model`), the variables ``prcp``, ``pet`` (``snow`` and ``temp``, optionally) are
            unavailable and replaced by ``sparse_prcp``, ``sparse_pet`` (``sparse_snow`` and ``sparse_temp``,
            optionally) and vice versa if the sparse_storage option has not been chosen.

        Access to a specific gauge mean precipitation time serie

        >>> model.mesh.code
        array(['V3524010', 'V3515010', 'V3517010'], dtype='<U8')
        >>> ind = np.argwhere(model.mesh.code == "V3524010").item()
        >>> ind
        0
        >>> model.atmos_data.mean_prcp[ind, :]
        array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)

        Access to a specific time step precipitation grid

        >>> time_step = 1200
        >>> model.atmos_data.prcp[..., time_step]
        array([[4.6       , 4.7000003 , 4.5       , 4.3       , 4.4       ,
                4.2000003 , 4.1       , 3.7       , 3.7       , 3.8       ,
                ...
                0.90000004, 0.8       , 0.6       , 0.4       , 0.4       ,
                0.1       , 0.1       , 0.1       ]], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.atmos_data.<TAB>
        model.atmos_data.copy()                    model.atmos_data.mean_temp
        model.atmos_data.from_handle(              model.atmos_data.pet
        model.atmos_data.init_array_sparse_pet()   model.atmos_data.prcp
        model.atmos_data.init_array_sparse_prcp()  model.atmos_data.snow
        model.atmos_data.init_array_sparse_snow()  model.atmos_data.sparse_pet
        model.atmos_data.init_array_sparse_temp()  model.atmos_data.sparse_prcp
        model.atmos_data.mean_pet                  model.atmos_data.sparse_snow
        model.atmos_data.mean_prcp                 model.atmos_data.sparse_temp
        model.atmos_data.mean_snow                 model.atmos_data.temp
        """

        return self._input_data.atmos_data

    @atmos_data.setter
    def atmos_data(self, value: Atmos_DataDT):
        self._input_data.atmos_data = value

    @property
    def rr_parameters(self) -> RR_ParametersDT:
        """
        Model rainfall-runoff parameters.

        Returns
        -------
        rr_parameters : `RR_ParametersDT <fcore._mwd_rr_parameters.RR_ParametersDT>`
            It returns a Fortran derived type containing the variables relating to the rainfall-runoff
            parameters.

        See Also
        --------
        Model.get_rr_parameters : Get the values of a Model rainfall-runoff parameter.
        Model.set_rr_parameters : Set the values of a Model rainfall-runoff parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model rainfall-runoff parameters

        >>> model.rr_parameters
        RR_ParametersDT
            keys: array(['ci', 'cp', 'ct', 'kexc', 'llr'], dtype='<U4')
            values: array([[[1.0e-06, 2.0e+02, 5.0e+02, 0.0e+00, 5.0e+00],
                [1.0e-06, 2.0e+02, 5.0e+02, 0.0e+00, 5.0e+00],
                [1.0e-06, 2.0e+02, 5.0e+02, 0.0e+00, 5.0e+00],
                ...
                [1.0e-06, 2.0e+02, 5.0e+02, 0.0e+00, 5.0e+00]]], dtype=float32)

        Access to a specific rainfall-runoff parameter grid with the getter method
        `get_rr_parameters <Model.get_rr_parameters>`

        >>> model.get_rr_parameters("cp")
        array([[200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                ...
                200., 200., 200., 200., 200., 200.]], dtype=float32)

        Set a value to a specific rainfall-runoff parameter grid with the setter method
        `set_rr_parameters <Model.set_rr_parameters>`

        >>> model.set_rr_parameters("cp", 273)
        >>> model.get_rr_parameters("cp")
        array([[273., 273., 273., 273., 273., 273., 273., 273., 273., 273., 273.,
                273., 273., 273., 273., 273., 273., 273., 273., 273., 273., 273.,
                ...
                273., 273., 273., 273., 273., 273.]], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.rr_parameters.<TAB>
        model.rr_parameters.copy()        model.rr_parameters.keys
        model.rr_parameters.from_handle(  model.rr_parameters.values
        """

        return self._parameters.rr_parameters

    @rr_parameters.setter
    def rr_parameters(self, value: RR_ParametersDT):
        self._parameters.rr_parameters = value

    @property
    def rr_initial_states(self) -> RR_StatesDT:
        """
        Model rainfall-runoff initial states.

        Returns
        -------
        rr_initial_states : `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>`
            It returns a Fortran derived type containing the variables relating to the rainfall-runoff
            initial states.

        See Also
        --------
        Model.get_rr_initial_states : Get the values of a Model rainfall-runoff initial state.
        Model.set_rr_initial_states : Set the values of a Model rainfall-runoff initial state.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model rainfall-runoff initial states

        >>> model.rr_initial_states
        RR_States
            keys: array(['hi', 'hp', 'ht', 'hlr'], dtype='<U3')
            values: array([[[1.e-02, 1.e-02, 1.e-02, 1.e-06],
                [1.e-02, 1.e-02, 1.e-02, 1.e-06],
                [1.e-02, 1.e-02, 1.e-02, 1.e-06],
                ...
                [1.e-02, 1.e-02, 1.e-02, 1.e-06]]], dtype=float32)

        Access to a specific rainfall-runoff initial state grid with the getter method
        `get_rr_initial_states <Model.get_rr_initial_states>`

        >>> model.get_rr_initial_states("hp")
        array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                ...
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32)

        Set a value to a specific rainfall-runoff initial state grid with the setter method
        `set_rr_initial_states <Model.set_rr_initial_states>`

        >>> model.set_rr_initial_states("hp", 0.29)
        >>> model.get_rr_initial_states("hp")
        array([[0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29,
                0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29,
                ...
                0.29, 0.29, 0.29, 0.29, 0.29, 0.29]], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.rr_initial_states.<TAB>
        model.rr_initial_states.copy()        model.rr_initial_states.keys
        model.rr_initial_states.from_handle(  model.rr_initial_states.values
        """

        return self._parameters.rr_initial_states

    @rr_initial_states.setter
    def rr_initial_states(self, value: RR_StatesDT):
        self._parameters.rr_initial_states = value

    @property
    def serr_mu_parameters(self) -> SErr_Mu_ParametersDT:
        """
        Model structural error mu parameters.

        Returns
        -------
        serr_mu_parameters : `SErr_Mu_ParametersDT <fcore._mwd_serr_mu_parameters.SErr_Mu_ParametersDT>`
            It returns a Fortran derived type containing the variables relating to the structural error mu
            parameters.

        See Also
        --------
        Model.get_serr_mu_parameters : Get the values of a Model structural error mu parameter.
        Model.set_serr_mu_parameters : Set the values of a Model structural error mu parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the structural error mu mapping to ``'Linear'`` (see `Model`). Default value in the
        ``Cance`` dataset is ``'Zero'`` (equivalent to no mu mapping)

        >>> setup["serr_mu_mapping"] = "Linear"
        >>> model = smash.Model(setup, mesh)

        Access to Model structural error mu parameters

        >>> model.serr_mu_parameters
        SErr_Mu_ParametersDT
            keys: array(['mg0', 'mg1'], dtype='<U3')
            values: array([[0., 0.],
               [0., 0.],
               [0., 0.]], dtype=float32)

        .. note::
            If we had left the default structural error mu mapping to ``'Zero'`` this is the output we
            would have obtained

            >>> model.serr_mu_parameters
            SErr_Mu_ParametersDT
                keys: array([], dtype=float64)
                values: array([], shape=(3, 0), dtype=float32)

        Access to a specific structural error mu parameter vector with the getter method
        `get_serr_mu_parameters <Model.get_serr_mu_parameters>`

        >>> model.get_serr_mu_parameters("mg0")
        array([0., 0., 0.], dtype=float32)

        Set a value to a specific structural error mu parameter vector with the setter method
        `set_serr_mu_parameters <Model.set_serr_mu_parameters>`

        >>> model.set_serr_mu_parameters("mg0", 11)
        >>> model.get_serr_mu_parameters("mg0")
        array([11., 11., 11.], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.serr_mu_parameters.<TAB>
        model.serr_mu_parameters.copy()        model.serr_mu_parameters.keys
        model.serr_mu_parameters.from_handle(  model.serr_mu_parameters.values
        """

        return self._parameters.serr_mu_parameters

    @serr_mu_parameters.setter
    def serr_mu_parameters(self, value: SErr_Mu_ParametersDT):
        self._parameters.serr_mu_parameters = value

    @property
    def serr_sigma_parameters(self) -> SErr_Sigma_ParametersDT:
        """
        Model structural error sigma parameters.

        Returns
        -------
        serr_sigma_parameters : `SErr_Sigma_ParametersDT <fcore._mwd_serr_sigma_parameters.SErr_Sigma_ParametersDT>`
            It returns a Fortran derived type containing the variables relating to the structural error sigma
            parameters.

        See Also
        --------
        Model.get_serr_sigma_parameters : Get the values of a Model structural error sigma parameter.
        Model.set_serr_sigma_parameters : Set the values of a Model structural error sigma parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model structural error sigma parameters

        >>> model.serr_sigma_parameters
        SErr_Sigma_ParametersDT
            keys: array(['sg0', 'sg1'], dtype='<U3')
            values: array([[1. , 0.2],
               [1. , 0.2],
               [1. , 0.2]], dtype=float32)

        Access to a specific structural error sigma parameter vector with the getter method
        `get_serr_sigma_parameters <Model.get_serr_sigma_parameters>`

        >>> model.get_serr_sigma_parameters("sg0")
        array([1., 1., 1.], dtype=float32)

        Set a value to a specific structural error sigma parameter vector with the setter method
        `set_serr_sigma_parameters <Model.set_serr_sigma_parameters>`

        >>> model.set_serr_sigma_parameters("sg0", 5.4)
        >>> model.get_serr_sigma_parameters("sg0")
        array([5.4, 5.4, 5.4], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.serr_sigma_parameters.<TAB>
        model.serr_sigma_parameters.copy()        model.serr_sigma_parameters.keys
        model.serr_sigma_parameters.from_handle(  model.serr_sigma_parameters.values
        """  # noqa: E501

        return self._parameters.serr_sigma_parameters

    @serr_sigma_parameters.setter
    def serr_sigma_parameters(self, value: SErr_Sigma_ParametersDT):
        self._parameters.serr_sigma_parameters = value

    @property
    def nn_parameters(self) -> NN_ParametersDT:
        """
        The weight and bias of the parameterization neural network.

        The neural network is used in hybrid model structures to correct internal fluxes.

        Returns
        -------
        nn_parameters : `NN_ParametersDT <fcore._mwd_nn_parameters.NN_ParametersDT>`
            It returns a Fortran derived type containing the weight and bias of the parameterization
            neural network.

        See Also
        --------
        Model.get_nn_parameters_weight : Get the weight of the parameterization neural network.
        Model.get_nn_parameters_bias : Get the bias of the parameterization neural network.
        Model.set_nn_parameters_weight : Set the values of the weight in the parameterization neural network.
        Model.set_nn_parameters_bias : Set the values of the bias in the parameterization neural network.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the hydrological module to ``'gr4_mlp'`` (hybrid hydrological model with multilayer
        perceptron)

        >>> setup["hydrological_module"] = "gr4_mlp"
        >>> model = smash.Model(setup, mesh)

        By default, the weight and bias of the parameterization neural network are set to zero.
        Access to their values with the getter method
        `get_nn_parameters_weight <Model.get_nn_parameters_weight>` or
        `get_nn_parameters_bias <Model.get_nn_parameters_bias>`

        >>> model.get_nn_parameters_bias()
        [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                dtype=float32), array([0., 0., 0., 0.], dtype=float32)]

        The output contains a list of weight or bias values for trainable layers.

        Set random values with the setter methods
        `set_nn_parameters_weight <Model.set_nn_parameters_weight>` or
        `set_nn_parameters_bias <Model.set_nn_parameters_bias>` using available initializers

        >>> model.set_nn_parameters_bias(initializer="uniform", random_state=0)
        >>> model.get_nn_parameters_bias()
        [array([ 0.09762701,  0.43037874,  0.20552675,  0.08976637, -0.1526904 ,
                0.29178822, -0.12482557,  0.78354603,  0.92732555, -0.23311697,
                0.5834501 ,  0.05778984,  0.13608912,  0.85119325, -0.85792786,
                -0.8257414 ], dtype=float32),
        array([-0.9595632 ,  0.6652397 ,  0.5563135 ,  0.74002427], dtype=float32)]

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.nn_parameters.<TAB>
        model.nn_parameters.bias_1                  model.nn_parameters.from_handle(
        model.nn_parameters.bias_2                  model.nn_parameters.weight_1
        model.nn_parameters.bias_3                  model.nn_parameters.weight_2
        model.nn_parameters.copy()                  model.nn_parameters.weight_3

        .. note::
            Not all layer weights and biases are used in the neural network.
            The default network only uses 2 layers, which means that ``weight_3`` and ``bias_3``
            are not used and are empty arrays in this case

            >>> model.nn_parameters.weight_3.size, model.nn_parameters.bias_3.size
            (0, 0)

        To set another neural network structure

        >>> setup["hidden_neuron"] = (32, 16)
        >>> model_2 = smash.Model(setup, mesh)

        In this case, the number of layers is 3 instead of 2

        >>> weights = model_2.get_nn_parameters_weight()
        >>> len(weights)
        3

        >>> weights[2].size
        64
        """

        return self._parameters.nn_parameters

    @nn_parameters.setter
    def nn_parameters(self, value: NN_ParametersDT):
        self._parameters.nn_parameters = value

    @property
    def response(self) -> ResponseDT:
        """
        Model response.

        Returns
        -------
        response : `ResponseDT <fcore._mwd_response.ResponseDT>`
            It returns a Fortran derived type containing the variables relating to the response.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model response

        >>> model.response
        ResponseDT
            q: array([[-99., -99., -99., ..., -99., -99., -99.],
               [-99., -99., -99., ..., -99., -99., -99.],
               [-99., -99., -99., ..., -99., -99., -99.]], dtype=float32)

        Run the direct Model to generate discharge responses

        >>> model.forward_run()
        </> Forward Run

        Access to a specific gauge simulated discharge time serie

        >>> model.mesh.code
        array(['V3524010', 'V3515010', 'V3517010'], dtype='<U8')
        >>> ind = np.argwhere(model.mesh.code == "V3524010").item()
        >>> ind
        0
        >>> model.response.q[ind, :]
        array([1.9826430e-03, 1.3466669e-07, 6.7617895e-12, ..., 2.2796249e+01,
               2.2655941e+01, 2.2517307e+01], dtype=float32)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.response.<TAB>
        model.response.copy()        model.response.q
        model.response.from_handle(
        """

        return self._output.response

    @response.setter
    def response(self, value: ResponseDT):
        self._output.response = value

    @property
    def rr_final_states(self) -> RR_StatesDT:
        """
        Model rainfall-runoff final states.

        Returns
        -------
        rr_final_states : `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>`
            It returns a Fortran derived type containing the variables relating to the rainfall-runoff final
            states.

        See Also
        --------
        Model.get_rr_final_states : Get the values of a Model rainfall-runoff final state.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to Model rainfall-runoff final states

        >>> model.rr_final_states
        RR_StatesDT
            keys: array(['hi', 'hp', 'ht', 'hlr'], dtype='<U3')
            values: array([[[-99., -99., -99., -99.],
                [-99., -99., -99., -99.],
                ...
                [-99., -99., -99., -99.]]], dtype=float32)

        Run the direct Model to generate rainfall-runoff final states

        >>> model.forward_run()
        </> Forward Run

        Access to a specific rainfall-runoff final state grid with the getter method
        `get_rr_final_states <Model.get_rr_final_states>`

        >>> model.get_rr_final_states("hp")
        array([[-99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                  0.8682228 ,   0.88014543, -99.        , -99.        ,
                ...
                -99.        , -99.        , -99.        , -99.        ]],
              dtype=float32)

        .. note::
            Unlike rainfall-runoff initial states, there is no setter for rainfall-runoff final states.
            They are generated after any kind of simulation (i.e., forward_run, optimize, ...)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods

        >>> model.rr_final_states.<TAB>
        model.rr_final_states.copy()        model.rr_final_states.keys
        model.rr_final_states.from_handle(  model.rr_final_states.values
        """

        return self._output.rr_final_states

    @rr_final_states.setter
    def rr_final_states(self, value: RR_StatesDT):
        self._output.rr_final_states = value

    def copy(self) -> Model:
        """
        Create a deep copy of Model.

        Returns
        -------
        model : `Model`
            It returns a deep copy of Model.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Create a shallow copy of Model

        >>> model_sc = model

        Access to the rainfall-runoff parameter grid ``'cp'`` of the shallow copy

        >>> model_sc.get_rr_parameters("cp")
        array([[200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                ...
                200., 200., 200., 200., 200., 200.]], dtype=float32)

        Change the values of the rainfall-runoff parameter grid ``'cp'`` of the initial Model

        >>> model.set_rr_parameters("cp", 63)

        View the result on the shallow copy

        >>> model_sc.get_rr_parameters("cp")
        array([[63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                ...
                63., 63.]], dtype=float32)

        Create a deep copy of Model

        >>> model_dc = model.copy()

        Access to the rainfall-runoff parameter grid ``'cp'`` of the deep copy

        >>> model_dc.get_rr_parameters("cp")
        array([[63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                ...
                63., 63.]], dtype=float32)

        Change the values of the rainfall-runoff parameter grid ``'cp'`` of the initial Model

        >>> model.set_rr_parameters("cp", 362)

        View the result on the deep copy

        >>> model_dc.get_rr_parameters("cp")
        array([[63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                ...
                63., 63.]], dtype=float32)
        """

        return self.__copy__()

    def get_rr_parameters(self, key: str) -> NDArray[np.float32]:
        """
        Get the values of a Model rainfall-runoff parameter.

        Parameters
        ----------
        key : `str`
            The name of the rainfall-runoff parameter.

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(nrow, ncol)* representing the values of the rainfall-runoff parameter.

        See Also
        --------
        Model.rr_parameters : Model rainfall-runoff parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to a specific rainfall-runoff parameter grid

        >>> model.get_rr_parameters("cp")
        array([[200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                ...
                200., 200., 200., 200., 200., 200.]], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``rr_parameters.values`` array (as shown below)
            but is simpler to use

        Access the rainfall-runoff parameter keys

        >>> model.rr_parameters.keys
        array(['ci', 'cp', 'ct', 'kexc', 'llr'], dtype='<U4')

        Get the index of the rainfall-runoff parameter ``'cp'``

        >>> ind = np.argwhere(model.rr_parameters.keys == "cp").item()
        >>> ind
        1

        Slice the ``rr_parameters.values`` array on the last axis

        >>> model.rr_parameters.values[..., ind]
        array([[200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                ...
                200., 200., 200., 200., 200., 200.]], dtype=float32)
        """

        key = _standardize_get_rr_parameters_args(self, key)
        ind = np.argwhere(self._parameters.rr_parameters.keys == key).item()

        return self._parameters.rr_parameters.values[..., ind]

    def set_rr_parameters(self, key: str, value: Numeric | NDArray[Any]):
        """
        Set the values of a Model rainfall-runoff parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff parameter.

        value : `float` or `numpy.ndarray`
            The value(s) to set to the rainfall-runoff parameter.
            If the value is a `numpy.ndarray`, its shape must be broadcastable into the rainfall-runoff
            parameter shape.

        See Also
        --------
        Model.get_rr_parameters : Get the values of a Model rainfall-runoff parameter.
        Model.rr_parameters : Model rainfall-runoff parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Set a specific value to a rainfall-runoff parameter grid

        >>> model.set_rr_parameters("cp", 150)

        Access its value

        >>> model.get_rr_parameters("cp")
        array([[150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
                150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
                ...
                150, 150, 150, 150, 150, 150]], dtype=float32)

        Set a grid with a shape equivalent to the rainfall-runoff parameter filled with random values between
        ``10`` and ``500``

        Get the rainfall-runoff parameter shape

        >>> shape = model.get_rr_parameters("cp").shape
        >>> shape
        (28, 28)

        Generate the random grid

        >>> import numpy as np
        >>> np.random.seed(99)
        >>> random_arr = np.random.randint(10, 500, shape)
        >>> random_arr
        array([[139,  45, 195, 178, 211, 242, 220,  78, 207, 446, 395, 417, 264,
                301,  65, 459, 186, 231,  69, 496, 373, 254, 225, 140, 202, 150,
                ...
                107, 386]])

        Set to the rainfall-runoff parameter the random grid

        >>> model.set_rr_parameters("cp", random_arr)
        >>> model.get_rr_parameters("cp")
        array([[139.,  45., 195., 178., 211., 242., 220.,  78., 207., 446., 395.,
                417., 264., 301.,  65., 459., 186., 231.,  69., 496., 373., 254.,
                ...
                243., 424., 301., 413., 107., 386.]], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``rr_parameters.values`` array (as shown below)
            and change the values but is simpler and ``safer`` to use.

        Access the rainfall-runoff parameter keys

        >>> model.rr_parameters.keys
        array(['ci', 'cp', 'ct', 'kexc', 'llr'], dtype='<U4')

        Get the index of the rainfall-runoff parameter ``'cp'``

        >>> ind = np.argwhere(model.rr_parameters.keys == "cp").item()
        >>> ind
        1

        Slice the ``rr_parameters.values`` array on the last axis and change its values

        >>> model.rr_parameters.values[..., ind] = 56
        >>> model.rr_parameters.values[..., ind]
        array([[56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56.,
                56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56.,
                ...
                56., 56.]], dtype=float32)

        .. warning::
            In that case, there's no problem to set the value ``56`` to the rainfall-runoff parameter
            ``'cp'``, but each rainfall-runoff parameter has a feasibility domain, and that outside this
            domain, the model cannot run. For example, the feasibility domain of ``'cp'`` is
            :math:`]0, +\\inf[`.

        Trying to set a negative value to the rainfall-runoff parameter ``'cp'`` without the setter

        >>> model.rr_parameters.values[..., ind] = -47
        >>> model.rr_parameters.values[..., ind]
        array([[-47., -47., -47., -47., -47., -47., -47., -47., -47., -47., -47.,
                -47., -47., -47., -47., -47., -47., -47., -47., -47., -47., -47.,
                ...
                -47., -47., -47., -47., -47., -47.]], dtype=float32)

        No particular problem doing this but trying with the setter

        >>> model.set_rr_parameters("cp", -47)
        ValueError: Invalid value for model rr_parameter 'cp'. rr_parameter domain [-47, -47] is not included
        in the feasible domain ]0, inf[

        Finally, trying to run the Model with a negative value set to the rainfall-runoff parameter ``'cp'``
        leads to the same error.

        >>> model.forward_run()
        ValueError: Invalid value for model rr_parameter 'cp'. rr_parameter domain [-47, -47] is not included
        in the feasible domain ]0, inf[
        """

        key, value = _standardize_set_rr_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.rr_parameters.keys == key).item()

        self._parameters.rr_parameters.values[..., ind] = value

    def get_rr_initial_states(self, key: str) -> NDArray[np.float32]:
        """
        Get the values of a Model rainfall-runoff initial state.

        Parameters
        ----------
        key : `str`
            The name of the rainfall-runoff initial state.

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(nrow, ncol)* representing the values of the rainfall-runoff initial state.

        See Also
        --------
        Model.rr_initial_states : Model rainfall-runoff initial states.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to a specific rainfall-runoff initial state grid

        >>> model.get_rr_initial_states("hp")
        array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                ...
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``rr_initial_states.values`` array (as shown
            below) but is simpler to use

        Access the rainfall-runoff state keys

        >>> model.rr_initial_states.keys
        array(['hi', 'hp', 'ht', 'hlr'], dtype='<U3')

        Get the index of the rainfall-runoff initial state ``'hp'``

        >>> ind = np.argwhere(model.rr_initial_states.keys == "hp").item()
        >>> ind
        1

        Slice the ``rr_initial_states.values`` array on the last axis

        >>> model.rr_initial_states.values[..., ind]
        array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                ...
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32)
        """

        key = _standardize_get_rr_initial_states_args(self, key)
        ind = np.argwhere(self._parameters.rr_initial_states.keys == key).item()

        return self._parameters.rr_initial_states.values[..., ind]

    def set_rr_initial_states(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values of a Model rainfall-runoff initial state.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff initial state.

        value : `float` or `numpy.ndarray`
            The value(s) to set to the rainfall-runoff initial state.
            If the value is a `numpy.ndarray`, its shape must be broadcastable into the rainfall-runoff
            initial state shape.

        See Also
        --------
        Model.get_rr_initial_states : Get the values of a Model rainfall-runoff initial state.
        Model.rr_initial_states : Model rainfall-runoff initial states.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Set a specific value to a rainfall-runoff initial state grid

        >>> model.set_rr_initial_states("hp", 0.22)

        Access its value

        >>> model.get_rr_initial_states("hp")
        array([[0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
                0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
                ...
                0.22, 0.22, 0.22, 0.22, 0.22, 0.22]], dtype=float32)

        Set a grid with a shape equivalent to the rainfall-runoff initial state filled with random values
        between ``0`` and ``1``

        Get the rainfall-runoff initial state shape

        >>> shape = model.get_rr_initial_states("hp").shape
        >>> shape
        (28, 28)

        Generate the random grid

        >>> import numpy as np
        >>> np.random.seed(99)
        >>> random_arr = np.random.rand(*shape)
        >>> random_arr
        array([[6.72278559e-01, 4.88078399e-01, 8.25495174e-01, 3.14463876e-02,
                8.08049963e-01, 5.65617420e-01, 2.97622499e-01, 4.66957205e-02,
                ...
                8.83966213e-01, 3.73980927e-01, 2.98742432e-01, 8.37281270e-01]])

        Set to the rainfall-runoff initial state the random grid

        >>> model.set_rr_initial_states("hp", random_arr)
        >>> model.get_rr_initial_states("hp")
        array([[6.72278559e-01, 4.88078399e-01, 8.25495174e-01, 3.14463876e-02,
                8.08049963e-01, 5.65617420e-01, 2.97622499e-01, 4.66957205e-02,
                ...
                8.83966213e-01, 3.73980927e-01, 2.98742432e-01, 8.37281270e-01]],
              dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``rr_initial_states.values`` array (as shown
            below) and change the values but is simpler and ``safer`` to use

        Access the rainfall-runoff initial state keys

        >>> model.rr_initial_states.keys
        array(['hi', 'hp', 'ht', 'hlr'], dtype='<U3')

        Get the index of the rainfall-runoff initial state ``'hp'``

        >>> ind = np.argwhere(model.rr_initial_states.keys == "hp").item()
        >>> ind
        1

        Slice the ``rr_initial_states.values`` array on the last axis and change its values

        >>> model.rr_initial_states.values[..., ind] = 0.56
        >>> model.rr_initial_states.values[..., ind]
        array([[0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56,
                0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56,
                ...
                0.56, 0.56, 0.56, 0.56, 0.56, 0.56]], dtype=float32)

        .. warning::
            In that case, there's no problem to set the value ``0.56`` to the rainfall-runoff initial state
            ``'hp'``, but each rainfall-runoff initial state has a feasibility domain, and that outside this
            domain, the model cannot run. For example, the feasibility domain of ``'hp'`` is :math:`]0, 1[`.

        Trying to set a value greater than 1 to the rainfall-runoff initial state ``'hp'`` without the setter

        >>> model.rr_initial_states.values[..., ind] = 21
        >>> model.rr_initial_states.values[..., ind]
        array([[21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21.,
                21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21.,
                ...
                21., 21., 21., 21., 21., 21.]], dtype=float32)

        No particular problem doing this but trying with the setter

        >>> model.set_rr_initial_states("hp", 21)
        ValueError: Invalid value for model rr_initial_state 'hp'. rr_initial_state domain [21, 21] is not
        included in the feasible domain ]0, 1[

        Finally, trying to run the Model with a value greater than 1 set to the rainfall-runoff initial
        state ``'hp'`` leads to the same error

        >>> model.forward_run()
        ValueError: Invalid value for model rr_initial_state 'hp'. rr_initial_state domain [21, 21] is not
        included in the feasible domain ]0, 1[
        """

        key, value = _standardize_set_rr_initial_states_args(self, key, value)
        ind = np.argwhere(self._parameters.rr_initial_states.keys == key).item()

        self._parameters.rr_initial_states.values[..., ind] = value

    def get_serr_mu_parameters(self, key: str) -> NDArray[np.float32]:
        """
        Get the values of a Model structural error mu parameter.

        Parameters
        ----------
        key : `str`
            The name of the structural error mu parameter.

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(ng,)* representing the values of the structural error mu parameter.

        See Also
        --------
        Model.serr_mu_parameters : Model structural error mu parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the structural error mu mapping to ``'Linear'`` (see `Model`). Default value in the
        ``Cance`` dataset is ``'Zero'`` (equivalent to no mu mapping)

        >>> setup["serr_mu_mapping"] = "Linear"
        >>> model = smash.Model(setup, mesh)

        Access to a specific structural error mu parameter vector

        >>> model.get_serr_mu_parameters("mg0")
        array([0., 0., 0.], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``serr_mu_parameters.values`` array (as shown
            below) but is simpler to use.

        Access the structural error mu parameter keys

        >>> model.serr_mu_parameters.keys
        array(['mg0', 'mg1'], dtype='<U3')

        Get the index of the structural error mu parameter ``'mg0'``

        >>> ind = np.argwhere(model.serr_mu_parameters.keys == "mg0").item()
        >>> ind
        0

        Slice the ``serr_mu_parameters.values`` array on the last axis

        >>> model.serr_mu_parameters.values[..., ind]
        array([0., 0., 0.], dtype=float32)
        """

        key = _standardize_get_serr_mu_parameters_args(self, key)
        ind = np.argwhere(self._parameters.serr_mu_parameters.keys == key).item()

        return self._parameters.serr_mu_parameters.values[..., ind]

    def set_serr_mu_parameters(self, key: str, value: Numeric | NDArray[Any]):
        """
        Set the values of a Model structural error mu parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the structural error mu parameter.

        value : `float` or `numpy.ndarray`
            The value(s) to set to the structural error mu parameter.
            If the value is a `numpy.ndarray`, its shape must be broadcastable into the structural error mu
            parameter shape.

        See Also
        --------
        Model.get_serr_mu_parameters : Get the values of a Model structural error mu parameter.
        Model.serr_mu_parameters : Model structural error mu parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the structural error mu mapping to ``'Linear'`` (see `Model`). Default value in the
        ``Cance`` dataset is ``'Zero'`` (equivalent to no mu mapping)

        >>> setup["serr_mu_mapping"] = "Linear"
        >>> model = smash.Model(setup, mesh)

        Set a specific value to a structural error mu parameter vector

        >>> model.set_serr_mu_parameters("mg0", 10)

        Access its value

        >>> model.get_serr_mu_parameters("mg0")
        array([10., 10., 10.], dtype=float32)

        Set a vector with a shape equivalent to the structural error mu parameter

        Get the structural error mu parameter size (equivalent to the number of gauge ``model.mesh.ng``)

        >>> size = model.get_serr_mu_parameters("mg0").size
        >>> size
        3

        Generate the vector

        >>> vec = np.arange(1, size + 1)
        >>> vec
        array([1, 2, 3])

        Set to the structural error mu parameter the vector

        >>> model.set_serr_mu_parameters("mg0", vec)
        >>> model.get_serr_mu_parameters("mg0")
        array([1., 2., 3.], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``serr_mu_parameters.values`` array (as shown
            below) and change the values but is simpler and ``safer`` to use.

        Access the structual error mu parameter keys

        >>> model.serr_mu_parameters.keys
        array(['mg0', 'mg1'], dtype='<U3')

        Get the index of the structural error mu parameter ``'mg0'``

        >>> ind = np.argwhere(model.serr_mu_parameters.keys == "mg0").item()
        >>> ind
        0

        Slice the ``serr_mu_parameters.values`` array on the last axis and change its values

        >>> model.serr_mu_parameters.values[..., ind] = 24
        >>> model.serr_mu_parameters.values[..., ind]
        array([[24., 24., 24.]], dtype=float32)

        .. warning::
            In that case, there's no problem to set the value ``24`` to the structural error mu parameter
            ``'mg0'``, but each structural error mu parameter has a feasibility domain, and that outside this
            domain, the model cannot run. For example, the feasibility domain of ``'mg0'`` is
            :math:`]-\\inf, +\\inf[` so in this case, you should not have any problems.
        """

        key, value = _standardize_set_serr_mu_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.serr_mu_parameters.keys == key).item()

        self._parameters.serr_mu_parameters.values[..., ind] = value

    def get_serr_sigma_parameters(self, key: str) -> NDArray[np.float32]:
        """
        Get the values of a Model structural error sigma parameter.

        Parameters
        ----------
        key : `str`
            The name of the structural error sigma parameter.

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(ng,)* representing the values of the structural error sigma parameter.

        See Also
        --------
        Model.serr_sigma_parameters : Model structural error sigma parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Access to a specific structural error sigma parameter vector

        >>> model.get_serr_sigma_parameters("sg0")
        array([1., 1., 1.], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``serr_sigma_parameters.values`` array (as shown
            below) but is simpler to use.

        Access the structural error sigma parameter keys

        >>> model.serr_sigma_parameters.keys
        array(['sg0', 'sg1'], dtype='<U3')

        Get the index of the structural error sigma parameter ``'sg0'``

        >>> ind = np.argwhere(model.serr_sigma_parameters.keys == "sg0").item()
        >>> ind
        0

        Slice the ``serr_sigma_parameters.values`` array on the last axis

        >>> model.serr_sigma_parameters.values[..., ind]
        array([1., 1., 1.], dtype=float32)
        """

        key = _standardize_get_serr_sigma_parameters_args(self, key)
        ind = np.argwhere(self._parameters.serr_sigma_parameters.keys == key).item()

        return self._parameters.serr_sigma_parameters.values[..., ind]

    def set_serr_sigma_parameters(self, key: str, value: Numeric | NDArray[Any]):
        """
        Set the values of a Model structural error sigma parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the structural error sigma parameter.

        value : `float` or `numpy.ndarray`
            The value(s) to set to the structural error sigma parameter.
            If the value is a `numpy.ndarray`, its shape must be broadcastable into the structural error sigma
            parameter shape.

        See Also
        --------
        Model.get_serr_sigma_parameters : Get the values of a Model structural error sigma parameter.
        Model.serr_sigma_parameters : Model structural error sigma parameters.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Set a specific value to a structural error sigma parameter vector

        >>> model.set_serr_sigma_parameters("sg0", 2)

        Access its value

        >>> model.get_serr_sigma_parameters("sg0")
        array([2., 2., 2.], dtype=float32)

        Set a vector with a shape equivalent to the structural error sigma parameter

        Get the structural error sigma parameter size (equivalent to the number of gauge ``model.mesh.ng``)

        >>> size = model.get_serr_sigma_parameters("sg0").size
        >>> size
        3

        Generate the vector

        >>> vec = np.arange(1, size + 1)
        >>> vec
        array([1, 2, 3])

        Set to the structural error sigma parameter the vector

        >>> model.set_serr_sigma_parameters("sg0", vec)
        >>> model.get_serr_sigma_parameters("sg0")
        array([1., 2., 3.], dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``serr_sigma_parameters.values`` array (as shown
            below) and change the values but is simpler and ``safer`` to use.

        Access the structual error sigma parameter keys

        >>> model.serr_sigma_parameters.keys
        array(['sg0', 'sg1'], dtype='<U3')

        Get the index of the structural error sigma parameter ``'sg0'``

        >>> ind = np.argwhere(model.serr_sigma_parameters.keys == "sg0").item()
        >>> ind
        0

        Slice the ``serr_sigma_parameters.values`` array on the last axis and change its values

        >>> model.serr_sigma_parameters.values[..., ind] = 0.5
        >>> model.serr_sigma_parameters.values[..., ind]
        array([[0.5, 0.5, 0.5]], dtype=float32)

        .. warning::
            In that case, there's no problem to set the value ``0.5`` to the structural error sigma parameter
            ``'sg0'``,
            but each structural error sigma parameter has a feasibility domain, and that outside this domain,
            the model cannot run. For example, the feasibility domain of ``'sg0'`` is :math:`]0, +\\inf[`.

        Trying to set a negative value to the strutural error sigma parameter ``'sg0'`` without the setter

        >>> model.serr_sigma_parameters.values[..., ind] = -1
        >>> model.serr_sigma_parameters.values[..., ind]
        array([[-1., -1., -1.]], dtype=float32)

        No particular problem doing this but trying with the setter

        >>> model.set_serr_sigma_parameters("sg0", -1)
        ValueError: Invalid value for model serr_sigma_parameter 'sg0'. serr_sigma_parameter domain [-1, -1]
        is not included in the feasible domain ]0, inf[

        Finally, trying to run the Model with a negative value set to the structural error sigma parameter
        ``'sg0'`` leads to the same error

        >>> model.forward_run()
        ValueError: Invalid value for model serr_sigma_parameter 'sg0'. serr_sigma_parameter domain [-1, -1]
        is not included in the feasible domain ]0, inf[
        """

        key, value = _standardize_set_serr_sigma_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.serr_sigma_parameters.keys == key).item()

        self._parameters.serr_sigma_parameters.values[..., ind] = value

    def get_rr_final_states(self, key: str) -> NDArray[np.float32]:
        """
        Get the values of a Model rainfall-runoff final state.

        Parameters
        ----------
        key : `str`
            The name of the rainfall-runoff final state.

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(nrow, ncol)* representing the values of the rainfall-runoff final state.

        See Also
        --------
        Model.rr_final_states : Model rainfall-runoff final states.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Run the direct Model to generate discharge responses

        >>> model.forward_run()

        Access to a specific rainfall-runoff final state grid

        >>> model.get_rr_final_states("hp")
        array([[-99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                  0.8682228 ,   0.88014543, -99.        , -99.        ,
                ...
                -99.        , -99.        , -99.        , -99.        ]],
              dtype=float32)

        .. note::
            This method is equivalent to directly slicing the ``rr_final_states.values`` array (as shown
            below) but is simpler to use.

        Access the rainfall-runoff state keys

        >>> model.rr_final_states.keys
        array(['hi', 'hp', 'ht', 'hlr'], dtype='<U3')

        Get the index of the rainfall-runoff final state ``'hp'``

        >>> ind = np.argwhere(model.rr_final_states.keys == "hp").item()
        >>> ind
        1

        Slice the ``rr_final_states.values`` array on the last axis

        >>> model.rr_final_states.values[..., ind]
        array([[-99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                -99.        , -99.        , -99.        , -99.        ,
                  0.8682228 ,   0.88014543, -99.        , -99.        ,
                ...
                -99.        , -99.        , -99.        , -99.        ]],
              dtype=float32)
        """

        key = _standardize_get_rr_final_states_args(self, key)
        ind = np.argwhere(self._output.rr_final_states.keys == key).item()

        return self._output.rr_final_states.values[..., ind]

    def get_rr_parameters_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Get the boundary condition for the Model rainfall-runoff parameters.

        Returns
        -------
        bounds : `dict[str, tuple[float, float]]`
            A dictionary representing the boundary condition for each rainfall-runoff parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.get_rr_parameters_bounds()
        {'ci': (1e-06, 100.0), 'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0),
         'kexc': (-50, 50), 'llr': (1e-06, 1000.0)}

        .. note::
            This method allows you to find out the default bounds for the rainfall-runoff parameters.
            These bounds are used during optimization if they are not modified in the optimization method
            argument.
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_RR_PARAMETERS.items()
            if key in STRUCTURE_RR_PARAMETERS[self.setup.structure]
        }

    def get_rr_initial_states_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Get the boundary condition for the Model rainfall-runoff initial states.

        Returns
        -------
        bounds : `dict[str, tuple[float, float]]`
            A dictionary representing the boundary condition for each rainfall-runoff initial state.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.get_rr_initial_states_bounds()
        {'hi': (1e-06, 0.999999), 'hp': (1e-06, 0.999999),
         'ht': (1e-06, 0.999999), 'hlr': (1e-06, 1000.0)}

        .. note::
            This method allows you to find out the default bounds for the rainfall-runoff initial states.
            These bounds are used during optimization if they are not modified in the optimization method
            argument.
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_RR_INITIAL_STATES.items()
            if key in STRUCTURE_RR_STATES[self.setup.structure]
        }

    def get_serr_mu_parameters_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Get the boundary condition for the Model structural error mu parameters.

        Returns
        -------
        bounds : `dict[str, tuple[float, float]]`
            A dictionary representing the boundary condition for each structural error mu parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the structural error mu mapping to ``'Linear'`` (see `Model`). Default value in the
        ``Cance`` dataset is ``'Zero'`` (equivalent to no mu mapping).

        >>> setup["serr_mu_mapping"] = "Linear"
        >>> model = smash.Model(setup, mesh)

        >>> model.get_serr_mu_parameters_bounds()
        {'mg0': (-1000000.0, 1000000.0), 'mg1': (-1000000.0, 1000000.0)}

        .. note::
            This method allows you to find out the default bounds for the structural error mu parameters.
            These bounds are used during optimization if they are not modified in the optimization method
            argument.
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_SERR_MU_PARAMETERS.items()
            if key in SERR_MU_MAPPING_PARAMETERS[self.setup.serr_mu_mapping]
        }

    def get_serr_sigma_parameters_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Get the boundary condition for the Model structural error sigma parameters.

        Returns
        -------
        bounds : `dict[str, tuple[float, float]]`
            A dictionary representing the boundary condition for each structural error sigma parameter.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.get_serr_sigma_parameters_bounds()
        {'sg0': (1e-06, 1000.0), 'sg1': (1e-06, 10.0)}

        .. note::
            This method allows you to find out the default bounds for the structural error sigma parameters.
            These bounds are used during optimization if they are not modified in the optimization method
            argument.
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS.items()
            if key in SERR_SIGMA_MAPPING_PARAMETERS[self.setup.serr_sigma_mapping]
        }

    def get_serr_mu(self) -> NDArray[np.float32]:
        """
        Get the structural error mu value by applying the mu mapping.

        .. hint::
            See the :ref:`math_num_documentation.bayesian_estimation` section

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(ng, ntime_step)* representing the values of the structural error mu for each
            gauge and each time step.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the structural error mu mapping to ``'Linear'`` (see `Model`). Default value in the
        ``Cance`` dataset is ``'Zero'`` (equivalent to no mu mapping)

        >>> setup["serr_mu_mapping"] = "Linear"
        >>> model = smash.Model(setup, mesh)

        The structural error mu mapping is set to ``'Linear'``.
        Therefore, the mapping of mu parameters to mu is: :math:`\\mu(g,t)=\\mu_0(g)+\\mu_1(g)q(g,t)` with:

        - :math:`\\mu`, the mean of structural errors,

        - :math:`\\mu_0` and :math:`\\mu_1`, the structural error mu parameters with respect to
          ``'Linear'`` mapping,

        - :math:`q`, the model response (i.e., the discharge),

        - :math:`g` and :math:`t`, the index refering to the gauge and time step respectively

        Run the direct Model to generate discharge responses

        >>> model.forward_run()

        Set arbitrary values to structural error mu parameters

        >>> model.set_serr_mu_parameters("mg0", 1)
        >>> model.set_serr_mu_parameters("mg1", 2)

        Retrieve the mu value with the `get_serr_mu <Model.get_serr_mu>` method

        >>> mu = model.get_serr_mu()
        >>> mu
        array([[ 1.0039653,  1.0000002,  1.       , ..., 46.5925   , 46.311882 ,
                46.034615 ],
               [ 1.0004755,  1.       ,  1.       , ..., 10.65963  , 10.61587  ,
                10.572574 ],
               [ 1.0000595,  1.       ,  1.       , ...,  3.563775 ,  3.5520396,
                 3.5404253]], dtype=float32)

        This is equivalent to

        >>> mg0 = model.get_serr_mu_parameters("mg0").reshape(-1, 1)
        >>> mg1 = model.get_serr_mu_parameters("mg1").reshape(-1, 1)
        >>> q = model.response.q
        >>> mu2 = mg0 + mg1 * q
        >>> np.allclose(mu, mu2)
        True
        """

        serr_mu = np.zeros(shape=(self.mesh.ng, self.setup.ntime_step), order="F", dtype=np.float32)
        wrap_get_serr_mu(self.setup, self.mesh, self._parameters, self._output, serr_mu)
        return serr_mu

    def get_serr_sigma(self) -> NDArray[np.float32]:
        """
        Get the structural error sigma value by applying the sigma mapping.

        .. hint::
            See the :ref:`math_num_documentation.bayesian_estimation` section

        Returns
        -------
        value : `numpy.ndarray`
            An array of shape *(ng, ntime_step)* representing the values of the structural error sigma for
            each gauge and each time step.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        The structural error sigma mapping is set to ``'Linear'``

        >>> model.setup.serr_sigma_mapping
        'Linear'

        Therefore, the mapping of sigma parameters to sigma is:
        :math:`\\sigma(g,t)=\\sigma_0(g)+\\sigma_1(g)q(g,t)` with:

        - :math:`\\sigma`, the standard deviation of structural errors,

        - :math:`\\sigma_0` and :math:`\\sigma_1`, the structural error sigma parameters with respect to
          ``'Linear'`` mapping,

        - :math:`q`, the model response (i.e., the discharge),

        - :math:`g` and :math:`t`, the index refering to the gauge and time step respectively

        Run the direct Model to generate discharge responses

        >>> model.forward_run()

        Retrieve the sigma value with the `get_serr_sigma <Model.get_serr_sigma>` method

        >>> sigma = model.get_serr_sigma()
        >>> sigma
        array([[1.0003965, 1.       , 1.       , ..., 5.55925  , 5.5311885,
                5.5034614],
               [1.0000476, 1.       , 1.       , ..., 1.965963 , 1.9615871,
                1.9572574],
               [1.000006 , 1.       , 1.       , ..., 1.2563775, 1.255204 ,
                1.2540425]], dtype=float32)

        This is equivalent to

        >>> sg0 = model.get_serr_sigma_parameters("sg0").reshape(-1, 1)
        >>> sg1 = model.get_serr_sigma_parameters("sg1").reshape(-1, 1)
        >>> q = model.response.q
        >>> sigma2 = sg0 + sg1 * q
        >>> np.allclose(sigma, sigma2)
        True
        """
        serr_sigma = np.zeros(shape=(self.mesh.ng, self.setup.ntime_step), order="F", dtype=np.float32)
        wrap_get_serr_sigma(self.setup, self.mesh, self._parameters, self._output, serr_sigma)
        return serr_sigma

    def get_nn_parameters_weight(self) -> list[NDArray[np.float32]]:
        """
        Get the weight of the parameterization neural network.

        Returns
        -------
        value : list[`numpy.ndarray`]
            A list of arrays representing the weights of trainable layers.

        See Also
        --------
        Model.nn_parameters : The weight and bias of the parameterization neural network.
        Model.set_nn_parameters_weight : Set the values of the weight in the parameterization neural network.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the hydrological module to ``'gr4_mlp'`` (hybrid hydrological model with multilayer
        perceptron)

        >>> setup["hydrological_module"] = "gr4_mlp"

        Set the number of neurons in the hidden layer to 3 (the default value is 16, if not set)

        >>> setup["hidden_neuron"] = 3
        >>> model = smash.Model(setup, mesh)

        By default, the weights of trainable layers are set to zero.
        Access to their values with the getter methods
        `get_nn_parameters_weight <Model.get_nn_parameters_weight>`

        >>> model.get_nn_parameters_weight()
        [array([[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]], dtype=float32), array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]], dtype=float32)]

        The output contains a list of weight values for trainable layers.
        """

        return [
            getattr(self._parameters.nn_parameters, f"weight_{i + 1}") for i in range(self.setup.n_layers)
        ]

    def get_nn_parameters_bias(self) -> list[NDArray[np.float32]]:
        """
        Get the bias of the parameterization neural network.

        Returns
        -------
        value : list[`numpy.ndarray`]
            A list of arrays representing the biases of trainable layers.

        See Also
        --------
        Model.nn_parameters : The weight and bias of the parameterization neural network.
        Model.set_nn_parameters_bias : Set the values of the bias in the parameterization neural network.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the hydrological module to ``'gr4_mlp'`` (hybrid hydrological model with multilayer
        perceptron)

        >>> setup["hydrological_module"] = "gr4_mlp"

        Set the number of neurons in the hidden layer to 6 (the default value is 16, if not set)

        >>> setup["hidden_neuron"] = 6
        >>> model = smash.Model(setup, mesh)

        By default, the biases of trainable layers are set to zero.
        Access to their values with the getter methods
        `get_nn_parameters_bias <Model.get_nn_parameters_bias>`

        >>> model.get_nn_parameters_bias()
        [array([0., 0., 0., 0., 0., 0.], dtype=float32), array([0., 0., 0., 0.], dtype=float32)]

        The output contains a list of bias values for trainable layers.
        """

        return [getattr(self._parameters.nn_parameters, f"bias_{i + 1}") for i in range(self.setup.n_layers)]

    def set_nn_parameters_weight(
        self,
        value: list[NDArray[Any]] | None = None,
        initializer: str = "glorot_uniform",
        random_state: int | None = None,
    ):
        """
        Set the values of the weight in the parameterization neural network.

        Parameters
        ----------
        value : list[`float` or `numpy.ndarray`] or None, default None
            The list of values to set to the weights of trainable layers. If an element of the list is
            a `numpy.ndarray`, its shape must be broadcastable into the weight shape of that layer.
            If not used, a default or specified initialization method will be used.

        initializer : str, default 'glorot_uniform'
            Weight initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.
            Only used if **value** is not set.

        random_state : `int` or None, default None
            Random seed used for the initialization in case of using **initializer**.

            .. note::
                If not given, the neural network parameters will be initialized with a random seed.

        See Also
        --------
        Model.nn_parameters : The weight and bias of the parameterization neural network.
        Model.get_nn_parameters_weight : Get the weight of the parameterization neural network.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the hydrological module to ``'gr4_mlp'`` (hybrid hydrological model with multilayer
        perceptron)

        >>> setup["hydrological_module"] = "gr4_mlp"

        Set the number of neurons in the hidden layer to 3 (the default value is 16, if not set)

        >>> setup["hidden_neuron"] = 3
        >>> model = smash.Model(setup, mesh)

        Set random weights using Glorot uniform initializer

        >>> model.set_nn_parameters_weight(initializer="glorot_uniform", random_state=0)
        >>> model.get_nn_parameters_weight()
        [array([[ 0.09038505,  0.3984533 ,  0.1902808 ,  0.08310751],
                [-0.14136384,  0.27014342, -0.11556603,  0.7254226 ],
                [ 0.8585366 , -0.21582437,  0.54016984,  0.053503  ]], dtype=float32),
        array([[ 0.12599404,  0.78805184, -0.7942869 ],
                [-0.764488  , -0.8883829 ,  0.6158923 ],
                [ 0.51504624,  0.68512934,  0.886229  ],
                [ 0.55393404, -0.07132636,  0.5194391 ]], dtype=float32)]

        The output contains a list of weight values for trainable layers.

        Set weights with specified values

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model.set_nn_parameters_weight([0.01, np.random.normal(size=(4,3))])
        >>> model.get_nn_parameters_weight()
        [array([[0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01]], dtype=float32),
        array([[ 1.7640524 ,  0.4001572 ,  0.978738  ],
                [ 2.2408931 ,  1.867558  , -0.9772779 ],
                [ 0.95008844, -0.1513572 , -0.10321885],
                [ 0.41059852,  0.14404356,  1.4542735 ]], dtype=float32)]
        """

        value, initializer, random_state = _standardize_set_nn_parameters_weight_args(
            self, value, initializer, random_state
        )

        if value is None:
            if random_state is not None:
                np.random.seed(random_state)

            for i in range(self.setup.n_layers):
                (n_neuron, n_in) = getattr(self._parameters.nn_parameters, f"weight_{i + 1}").shape
                setattr(
                    self._parameters.nn_parameters,
                    f"weight_{i + 1}",
                    _initialize_nn_parameter(n_in, n_neuron, initializer),
                )

            # % Reset random seed if random_state is previously set
            if random_state is not None:
                np.random.seed(None)

        else:
            for i, val in enumerate(value):
                setattr(self._parameters.nn_parameters, f"weight_{i + 1}", val)

    def set_nn_parameters_bias(
        self,
        value: list[NDArray[Any]] | None = None,
        initializer: str = "zeros",
        random_state: int | None = None,
    ):
        """
        Set the values of the bias in the parameterization neural network.

        Parameters
        ----------
        value : list[`float` or `numpy.ndarray`] or None, default None
            The list of values to set to the biases of trainable layers. If an element of the list is
            a `numpy.ndarray`, its shape must be broadcastable into the bias shape of that layer.
            If not used, a default or specified initialization method will be used.

        initializer : str, default 'zeros'
            Bias initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.
            Only used if **value** is not set.

        random_state : `int` or None, default None
            Random seed used for the initialization in case of using **initializer**.

            .. note::
                If not given, the neural network parameters will be initialized with a random seed.

        See Also
        --------
        Model.nn_parameters : The weight and bias of the parameterization neural network.
        Model.get_nn_parameters_bias : Get the bias of the parameterization neural network.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        Set the hydrological module to ``'gr4_mlp'`` (hybrid hydrological model with multilayer
        perceptron)

        >>> setup["hydrological_module"] = "gr4_mlp"

        Set the number of neurons in the hidden layer to 6 (the default value is 16, if not set)

        >>> setup["hidden_neuron"] = 6
        >>> model = smash.Model(setup, mesh)

        Set random biases using Glorot normal initializer

        >>> model.set_nn_parameters_bias(initializer="glorot_normal", random_state=0)
        >>> model.get_nn_parameters_bias()
        [array([ 0.94292563,  0.21389303,  0.5231575 ,  1.1978078 ,  0.99825174, -0.522377  ],
            dtype=float32),
        array([ 0.60088867, -0.09572671, -0.06528133,  0.2596853 ],
            dtype=float32)]

        The output contains a list of bias values for trainable layers.

        Set biases with specified values

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model.set_nn_parameters_bias([np.random.normal(size=6), 0])
        >>> model.get_nn_parameters_bias()
        [array([ 1.7640524,  0.4001572,  0.978738 ,  2.2408931,  1.867558 ,
                -0.9772779], dtype=float32), array([0., 0., 0., 0.], dtype=float32)]
        """

        value, initializer, random_state = _standardize_set_nn_parameters_bias_args(
            self, value, initializer, random_state
        )

        if value is None:
            if random_state is not None:
                np.random.seed(random_state)

            for i in range(self.setup.n_layers):
                n_neuron = getattr(self._parameters.nn_parameters, f"bias_{i + 1}").shape[0]
                setattr(
                    self._parameters.nn_parameters,
                    f"bias_{i + 1}",
                    _initialize_nn_parameter(1, n_neuron, initializer).flatten(),
                )

            # % Reset random seed if random_state is previously set
            if random_state is not None:
                np.random.seed(None)

        else:
            for i, val in enumerate(value):
                setattr(self._parameters.nn_parameters, f"bias_{i + 1}", val)

    def adjust_interception(
        self,
        active_cell_only: bool = True,
    ):
        """
        Adjust the interception reservoir capacity.

        Parameters
        ----------
        active_cell_only : bool, default True
            If True, adjusts the interception capacity only for the active cells of the 2D spatial domain.
            If False, adjusts the interception capacity for all cells in the domain.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")

        By default, the interception capacity is automatically adjusted when the model is created.
        Now we set it to False and then manually adjust the interception capacity after model creation.

        >>> setup["adjust_interception"] = False
        >>> model = smash.Model(setup, mesh)

        >>> model.get_rr_parameters("ci")
        array([[1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                1.e-06, 1.e-06, 1.e-06, 1.e-06],
               ...
               [1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                   1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                   1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                   1.e-06, 1.e-06, 1.e-06, 1.e-06]], dtype=float32)

        Adjust the interception capacity for all cells in the spatial domain

        >>> model.adjust_interception(active_cell_only=False)

        >>> model.get_rr_parameters("ci")
        array([[1.        , 1.        , 1.        , 1.        , 1.        ,
                1.        , 1.        , 1.1       , 1.1       , 1.1       ,
                1.1       , 1.1       , 1.1       , 1.2       , 1.2       ,
                1.3000001 , 1.4       , 1.4       , 1.4       , 1.7       ,
                1.7       , 1.6       , 1.6       , 1.5       , 1.6       ,
                1.6       , 1.5       , 1.5       ],
               ...
               [1.        , 1.1       , 1.1       , 1.1       , 1.1       ,
                1.1       , 1.        , 1.1       , 1.1       , 1.1       ,
                1.2       , 1.1       , 1.1       , 1.1       , 1.1       ,
                1.2       , 1.2       , 1.2       , 1.3000001 , 1.3000001 ,
                1.3000001 , 1.3000001 , 1.3000001 , 1.3000001 , 1.3000001 ,
                1.4       , 1.4       , 1.5       ]], dtype=float32)
        """
        _adjust_interception(
            self.setup, self.mesh, self._input_data, self._parameters, active_cell_only=active_cell_only
        )

    @_model_forward_run_doc_substitution
    @_forward_run_doc_appender
    def forward_run(
        self,
        cost_options: dict[str, Any] | None = None,
        common_options: dict[str, Any] | None = None,
        return_options: dict[str, Any] | None = None,
    ) -> ForwardRun | None:
        args_options = [deepcopy(arg) for arg in [cost_options, common_options, return_options]]

        args = _standardize_forward_run_args(self, *args_options)

        return _forward_run(self, *args)

    @_model_optimize_doc_substitution
    @_optimize_doc_appender
    def optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict[str, Any] | None = None,
        cost_options: dict[str, Any] | None = None,
        common_options: dict[str, Any] | None = None,
        return_options: dict[str, Any] | None = None,
        callback: callable | None = None,
    ) -> Optimize | None:
        args_options = [
            deepcopy(arg) for arg in [optimize_options, cost_options, common_options, return_options]
        ]

        args = _standardize_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
            callback,
        )

        return _optimize(self, *args)

    @_set_control_optimize_doc_substitution
    @_set_control_optimize_doc_appender
    def set_control_optimize(
        self,
        control_vector: np.ndarray,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict[str, Any] | None = None,
    ):
        optimize_options = deepcopy(optimize_options)

        # % Only get mapping, optimizer, optimize_options and cost_options
        *args, _, _, _ = _standardize_optimize_args(
            self,
            mapping,
            optimizer,
            optimize_options,
            None,  # cost_options
            None,  # common_options
            None,  # return_options
            None,  # callback
        )

        # Cannot standardize 'control_vector' here before initializing model._parameters.control
        # it will be checked later after calling wrap_parameters_to_control
        _set_control(self, control_vector, *args)

    @_model_multiset_estimate_doc_substitution
    @_multiset_estimate_doc_appender
    def multiset_estimate(
        self,
        multiset: MultipleForwardRun,
        alpha: Numeric | ListLike | None = None,
        common_options: dict[str, Any] | None = None,
        return_options: dict[str, Any] | None = None,
    ) -> MultisetEstimate | None:
        args_options = [deepcopy(arg) for arg in [common_options, return_options]]

        args = _standardize_multiset_estimate_args(self, multiset, alpha, *args_options)

        return _multiset_estimate(self, *args)

    @_model_bayesian_optimize_doc_substitution
    @_bayesian_optimize_doc_appender
    def bayesian_optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict[str, Any] | None = None,
        cost_options: dict[str, Any] | None = None,
        common_options: dict[str, Any] | None = None,
        return_options: dict[str, Any] | None = None,
        callback: callable | None = None,
    ) -> BayesianOptimize | None:
        args_options = [
            deepcopy(arg) for arg in [optimize_options, cost_options, common_options, return_options]
        ]

        args = _standardize_bayesian_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
            callback,
        )

        return _bayesian_optimize(self, *args)

    @_set_control_bayesian_optimize_doc_substitution
    @_set_control_bayesian_optimize_doc_appender
    def set_control_bayesian_optimize(
        self,
        control_vector: np.ndarray,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict[str, Any] | None = None,
        cost_options: dict[str, Any] | None = None,
    ):
        args_options = [deepcopy(arg) for arg in [optimize_options, cost_options]]

        # % Only get mapping, optimizer, optimize_options and cost_options
        *args, _, _, _ = _standardize_bayesian_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
            None,  # common_options
            None,  # return_options
            None,  # callback
        )

        # Cannot standardize 'control_vector' here before initializing model._parameters.control
        # it will be checked later after calling wrap_parameters_to_control
        _set_control(self, control_vector, *args)
