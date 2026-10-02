"""The saved rows and the recorded events of a run, and the result built from them."""
import numpy as np

from Solverz.solvers.parser import parse_dae_v
from Solverz.solvers.solution import daesol

__all__ = []


class SolutionBuffer:
    """The rows saved during a run, as a list of times and a list of states.

    ``push`` keeps the state it is given, so the caller passes a copy. There is
    no capacity: ``to_daesol`` stacks the rows once at the end.
    """

    __slots__ = ('times', 'rows')

    def __init__(self):
        self.times = []
        self.rows = []

    def push(self, t, y):
        self.times.append(t)
        self.rows.append(y)

    @property
    def last_t(self):
        """The time of the last saved row, ``None`` before the first."""
        return self.times[-1] if self.times else None

    def __len__(self):
        return len(self.times)


class EventLog:
    """The recorded events of a run: time, state and component index.

    ``record`` keeps the state it is given; components found at one event
    time may share one state.
    """

    __slots__ = ('te', 'ye', 'ie')

    def __init__(self):
        self.te = []
        self.ye = []
        self.ie = []

    def record(self, t, y, i):
        self.te.append(t)
        self.ye.append(y)
        self.ie.append(i)

    def __len__(self):
        return len(self.te)


def to_daesol(sol, events, stats, address=None):
    """Build the legacy ``daesol`` from the saved rows and the recorded events.

    ``T`` and ``Y`` are new float64 arrays, ``Y`` C-contiguous of shape
    ``(len(T), n)``. ``te``, ``ye`` and ``ie`` are float64, float64 and int64,
    and all three are ``None`` when nothing was recorded, since ``parse_dae_v``
    cannot convert an empty ``ye``. With the ``Address`` of a ``Vars`` initial
    state, ``Y`` and ``ye`` are converted as ``dae_io_parser`` converts them.
    """
    T = np.array(sol.times, dtype=np.float64)
    Y = np.array(sol.rows, dtype=np.float64)
    if len(events) > 0:
        te = np.array(events.te, dtype=np.float64)
        ye = np.array(events.ye, dtype=np.float64)
        ie = np.array(events.ie, dtype=np.int64)
    else:
        te = ye = ie = None
    if address is not None:
        Y = parse_dae_v(Y, address)
        if te is not None:
            ye = parse_dae_v(ye, address)
    return daesol(T, Y, te, ye, ie, stats)
