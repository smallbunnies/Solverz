from typing import Callable, Optional, Union, List
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_array

from Solverz.num_api.Array import Array
from Solverz.variable.ssymbol import sSymBasic


class ParamBase:
    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number, csc_array] = None,
                 triggerable: bool = False,
                 trigger_var: Union[str, List[str]] = None,
                 trigger_fun: Callable = None,
                 dim: int = 1,
                 dtype=float,
                 sparse=False,
                 is_alias=False):
        # Time-varying sparse 2-D Params are unsupported by design.
        # Solverz's code generation — both the legacy ``MatVecMul``
        # path and the 0.8.1 ``Mat_Mul`` ``SolCF.csc_matvec`` fast
        # path — freezes a sparse ``dim=2`` param's CSC
        # decomposition (``<name>_data`` / ``_indices`` / ``_indptr``
        # / ``_shape0``) at model-build time. The mutable-matrix
        # Jacobian kernel caches its ``.data`` inside @njit scatter-
        # add loops, also at model-build time. A runtime trigger
        # update would silently be ignored by every downstream
        # consumer and produce wrong Newton steps.
        #
        # Reject at construction time so the error fires at the line
        # where the user wrote the offending declaration, rather
        # than deep inside ``FormJac``. ``TimeSeriesParam`` has the
        # same restriction (checked in its own ``__init__`` below,
        # since it always passes ``triggerable=False`` to this
        # constructor).
        if sparse and dim == 2 and triggerable:
            raise NotImplementedError(
                f"Parameter {name!r}: a sparse 2-D ``Param`` cannot "
                f"be declared ``triggerable=True``. Time-varying "
                f"sparse matrices are unsupported because Solverz's "
                f"code generation caches the matrix's CSC fields at "
                f"model-build time; a runtime trigger update would "
                f"silently be ignored. If you need a matrix whose "
                f"values change at runtime, rewrite the equation in "
                f"explicit element-wise form (one scalar ``Eqn`` per "
                f"row) using the per-row coefficients as 1-D "
                f"``Param`` or ``TimeSeriesParam`` instances, or use "
                f"a dense ``dim=2`` parameter (``sparse=False``) — "
                f"the fallback scipy path re-evaluates the full "
                f"expression on every call and tolerates updates."
            )
        self.name = name
        self.triggerable = triggerable
        self.trigger_var = [trigger_var] if isinstance(trigger_var, str) else trigger_var
        self.trigger_fun = trigger_fun
        self.dim = dim
        self.dtype = dtype
        self.sparse = sparse
        self.__v = None
        self.v = value
        self.is_alias = is_alias  # if the Param is an alias var

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value):

        if value is None:
            self.__v = None
        else:
            self.__v = Array(value, dim=self.dim, sparse=self.sparse, dtype=self.dtype)

    def get_v_t(self, t):
        return self.v

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"


class Param(ParamBase, sSymBasic):

    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number, csc_array] = None,
                 triggerable: bool = False,
                 trigger_var: Union[str, List[str]] = None,
                 trigger_fun: Callable = None,
                 dim: int = 1,
                 dtype=float,
                 sparse=False,
                 is_alias=False
                 ):
        ParamBase.__init__(self,
                           name,
                           value,
                           triggerable,
                           trigger_var,
                           trigger_fun,
                           dim,
                           dtype,
                           sparse,
                           is_alias)

        sSymBasic.__init__(self, name=name, Type='Para', value=value, dim=dim, sparse=sparse)


class IdxParam(ParamBase, sSymBasic):

    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None
                 ):
        ParamBase.__init__(self,
                           name,
                           value,
                           triggerable,
                           trigger_var,
                           trigger_fun,
                           dim=1,
                           dtype=int,
                           sparse=False)
        sSymBasic.__init__(self,
                           name=name,
                           Type='idx',
                           value=value,
                           dim=1)


class TimeSeriesParam(Param):
    def __init__(self,
                 name: str,
                 v_series,
                 time_series,
                 index=None,
                 value: Union[np.ndarray, list, Number] = None,
                 dim=1,
                 dtype=float,
                 sparse=False
                 ):
        # ``TimeSeriesParam`` is the other class of time-varying
        # parameter Solverz supports. The same immutability
        # constraint as in ``ParamBase.__init__`` applies to sparse
        # 2-D matrices: the generated module caches CSC fields at
        # build time and ``get_v_t(t)`` updates would be invisible
        # to the downstream fast path.
        if sparse and dim == 2:
            raise NotImplementedError(
                f"Parameter {name!r}: a sparse 2-D "
                f"``TimeSeriesParam`` is not supported. Solverz's "
                f"code generation caches the matrix's CSC fields at "
                f"model-build time; a time-series update at runtime "
                f"would silently be ignored. Use a 1-D "
                f"``TimeSeriesParam`` (scale / per-row coefficients) "
                f"and assemble the matrix element-wise, or use a "
                f"dense ``dim=2`` parameter (``sparse=False``)."
            )
        if value is None:
            value = v_series[0]
        super().__init__(name,
                         value,
                         triggerable=False,
                         trigger_var=None,
                         trigger_fun=None,
                         dim=dim,
                         dtype=dtype,
                         sparse=sparse)
        self.v_series = Array(v_series, dim=1)
        self.time_series = Array(time_series, dim=1)
        self.tend = self.time_series[-1]

        if len(self.v_series) != len(self.time_series):
            raise ValueError("Incompatible length between value series and time series!")
        if not np.all(np.diff(self.time_series) > 0):
            raise ValueError("Time stamp should be strictly monotonically increasing!")
        self.index = index
        self.vt = interp1d(self.time_series, self.v_series, kind='linear')

    def get_v_t(self, t):
        if t is None:
            return self.v

        if self.index is None:
            # No-index path: contract is a 1-D ndarray. Both
            # ``interp1d(scalar t).reshape((-1,))`` and
            # ``v_series[-1:]`` (a slice that keeps the leading axis)
            # are 1-D, so we return them directly.
            if t < self.tend:
                return self.vt(t).reshape((-1,))
            return self.v_series[-1:]

        # Index path: keep ``vt`` in its natural rank so it aligns with
        # ``temp[self.index]``'s slot — scalar index ↔ 0-D value,
        # array index ↔ 1-D value. numpy 2.4's strict rule against
        # ``a[scalar] = arr_of_size_1`` is what forced the split (the
        # old code reshaped to ``(-1,)`` unconditionally and tripped
        # the rule for the common ``index=int`` + 1-D ``v_series`` case).
        if t < self.tend:
            vt = self.vt(t)
        else:
            vt = self.v_series[-1]
        temp = self.v.copy()
        temp[self.index] = vt
        return temp

    def __repr__(self):
        return f"TimeSeriesParam: {self.name} value: {self.v}"
