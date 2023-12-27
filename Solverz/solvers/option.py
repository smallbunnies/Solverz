class Opt:
    def __init__(self,
                 atol=1e-6,
                 rtol=1e-3,
                 f_savety=0.9,
                 facmax=6,
                 fac1=0.2,
                 fac2=6,
                 scheme='rodas',
                 fix_h: bool = False,
                 hinit=None,
                 hmax=None):
        self.atol = atol
        self.rtol = rtol
        self.f_savety = f_savety
        self.facmax = facmax
        self.fac1 = fac1
        self.fac2 = fac2
        self.fix_h = fix_h  # To force the step sizes to be invariant. This is not robust.
        self.hinit = hinit
        self.hmax = hmax
        self.scheme = scheme
