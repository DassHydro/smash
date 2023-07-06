from __future__ import annotations

from typing import TYPE_CHECKING

from smash.signal_analysis.signatures._sign_computation import _sign_computation
from smash.signal_analysis.signatures._standardize import _standardize_signatures

if TYPE_CHECKING:
    from smash.core.model import Model


__all__ = ["Signatures", "signatures"]


class Signatures(dict):
    """
    Represents signatures computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    cont : dict
        Continuous signatures. The keys are

        - 'obs': a dataframe representing observation results.
        - 'sim': a dataframe representing simulation results.

        The column names of both dataframes consist of the catchment code and studied signature names.

    event : dict
        Flood event signatures. The keys are

        - 'obs': a dataframe representing observation results.
        - 'sim': a dataframe representing simulation results.

        The column names of both dataframes consist of the catchment code, the season that event occurrs, the beginning/end of each event and studied signature names.

    See Also
    --------
    Model.signatures: Compute continuous or/and flood event signatures of the Model.

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


def signatures(
    model: Model,
    sign: str | list | None = None,
    obs_comp: bool = True,
    sim_comp: bool = True,
    event_seg: dict | None = None,
):
    """
    Compute continuous or/and flood event signatures of the Model.

    .. hint::
        See the :ref:`User Guide <user_guide.in_depth.signatures.computation>` and :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.hydrological_signatures>` for more.

    Parameters
    ----------
    model : Model
        Model object.

    sign : str, list of str or None, default None
        Define signature(s) to compute. Should be one of

        - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures)
        - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures)

        .. note::
            If not given, all of continuous and flood event signatures will be computed.

    obs_comp : bool, default True
        If True, compute the signatures from observed discharges.

    sim_comp : bool, default True
        If True, compute the signatures from simulated discharges.

    event_seg : dict or None, default None
        A dictionary of event segmentation options when calculating flood event signatures. The keys are

        - 'peak_quant'
        - 'max_duration'

        See `smash.Model.event_segmentation` for more.

        .. note::
            If not given in case flood signatures are computed, the default values will be set for these parameters.

    Returns
    -------
    res : Signatures
        The signatures computation results represented as a `Signatures` object.

    See Also
    --------
    Signatures: Represents signatures computation result.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model.run(inplace=True)

    Compute all observed and simulated signatures:

    >>> res = model.signatures()
    >>> res.cont["obs"]  # observed continuous signatures
            code       Crc     Crchf  ...   Cfp50      Cfp90
    0  V3524010  0.516207  0.191349  ...  3.3225  42.631802
    1  V3515010  0.509180  0.147217  ...  1.5755  10.628400
    2  V3517010  0.514302  0.148364  ...  0.3235   2.776700

    [3 rows x 9 columns]

    >>> res.event["sim"]  # simulated flood event signatures
            code  season               start  ...  Elt         Epf
    0  V3524010  autumn 2014-11-03 03:00:00  ...    3  106.190651
    1  V3515010  autumn 2014-11-03 10:00:00  ...    0   21.160324
    2  V3517010  autumn 2014-11-03 08:00:00  ...    1   5.613392

    [3 rows x 12 columns]

    """

    cs, es = _standardize_signatures(sign)

    if event_seg is None:
        event_seg = {}

    res = _sign_computation(model, cs, es, obs_comp, sim_comp, **event_seg)

    return Signatures(res)
