from __future__ import annotations

from smash.core._event_segmentation import _mask_event

from smash.solver._mw_multiple_run import compute_multiple_run

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.core.generate_samples import SampleResult

import numpy as np
import pandas as pd

__all__ = ["MultipleRunResult"]


class MultipleRunResult(dict):
    """
    Represents the multiple run result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.
    This may have an additional attribute (``qsim``), which is not listed here,
    depending on the specific return values requested in the :meth:`Model.multiple_run` method.

    Attributes
    ----------
    cost : numpy.ndarray
        The cost value for each parameters set.

    See Also
    --------
    Model.multiple_run: Compute multiple forward run of the Model.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Get the boundary constraints of the Model parameters/states and generate a sample

    >>> problem = model.get_bound_constraints()
    >>> sample = smash.generate_samples(problem, n=200, random_state=99)

    Compute the multiple run

    >>> mtprr = model.multiple_run(sample, ncpu=4, return_qsim=True)

    Access cost and qsim values

    >>> mtprr.cost
    array([1.2327451e+00, 1.2367475e+00, 1.2227478e+00, 4.7788401e+00,
    ...
           1.2392160e+00, 1.2278881e+00, 7.5998020e-01, 1.1763511e+00],
          dtype=float32)

    >>> mtprr.cost.shape
    (200,)

    >>> mtprr.qsim
    array([[[1.01048472e-05, 9.94086258e-06, 4.78204456e-05, ...,
    ...
            [3.45186263e-01, 6.55560493e-02, 4.66010673e-03, ...,
             8.93603489e-02, 1.37046015e+00, 1.07397830e+00]]], dtype=float32)

    >>> mtprr.qsim.shape
    (3, 1440, 200)
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [
                    k.rjust(m) + ": " + repr(v)
                    for k, v in sorted(self.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _get_ind_parameters_states(instance: Model, sample: SampleResult) -> np.ndarray:
    ind_parameters_states = np.zeros(
        shape=sample._problem["num_vars"], dtype=np.int32, order="F"
    )
    n_parameters = instance.setup._parameters_name.size

    for i, name in enumerate(sample._problem["names"]):
        if name in instance.setup._parameters_name:
            ind = np.argwhere(instance.setup._parameters_name == name)
            # % Transform Python to Fortran index
            ind_parameters_states[i] = ind + 1
        # % Already check, must be states if not parameters
        else:
            ind = np.argwhere(instance.setup._states_name == name)
            # % Transform Python to Fortran index
            ind_parameters_states[i] = n_parameters + ind + 1

    return ind_parameters_states


def _multiple_run_message(instance: Model, sample: SampleResult):
    sp4 = " " * 4

    sample_names = sample._problem["names"]
    ncpu = instance.setup._ncpu
    jobs_fun = instance.setup._optimize.jobs_fun
    wjobs_fun = instance.setup._optimize.wjobs_fun
    parameters = [el for el in sample_names if el in instance.setup._parameters_name]
    states = [el for el in sample_names if el in instance.setup._states_name]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code)
        if instance.setup._optimize.wgauge[ind] != 0
    ]
    wgauge = np.array([el for el in instance.setup._optimize.wgauge if el > 0])

    ret = []
    ret.append(f"{sp4}Nsample: {sample.n_sample}")
    ret.append(f"Ncpu: {ncpu}")
    ret.append(f"Jobs function: [ {' '.join(jobs_fun)} ]")
    ret.append(f"wJobs function: [ {' '.join(wjobs_fun.astype('U'))} ]")
    ret.append(f"Np: {len(parameters)} [ {' '.join(parameters)} ]")
    ret.append(f"Ns: {len(states)} [ {' '.join(states)} ]")
    ret.append(f"Ng: {len(code)} [ {' '.join(code)} ]")
    ret.append(f"wg: {len(wgauge)} [ {' '.join(wgauge.astype('U'))} ]")

    print(f"\n{sp4}".join(ret) + "\n")


def _multiple_run(
    instance: Model,
    sample: SampleResult,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    ncpu: int,
    return_qsim: bool,
    verbose: bool,
):
    instance.setup._ncpu = ncpu
    instance.setup._optimize.verbose = verbose

    # % send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wjobs_fun = wjobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    ind_parameters_states = _get_ind_parameters_states(instance, sample)

    res_cost = np.zeros(
        shape=sample.n_sample,
        dtype=np.float32,
        order="F",
    )

    if return_qsim:
        res_qsim = np.zeros(
            shape=(instance.mesh.ng, instance.setup._ntime_step, sample.n_sample),
            dtype=np.float32,
            order="F",
        )
    else:
        res_qsim = np.empty(
            shape=3 * (0,),
            dtype=np.float32,
            order="F",
        )
    if verbose:
        _multiple_run_message(instance, sample)

    compute_multiple_run(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        instance.states,
        instance.output,
        sample.to_numpy(),
        ind_parameters_states,
        res_cost,
        res_qsim,
    )  # % Fortran subroutine mw_multiple_run

    ret_dict = {"cost": res_cost}

    if return_qsim:
        ret_dict.update({"qsim": res_qsim})

    return MultipleRunResult(ret_dict)
