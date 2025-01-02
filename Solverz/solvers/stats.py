class Stats:

    def __init__(self, scheme=None):
        self.scheme = scheme
        self.nstep = 0
        self.nfeval = 0
        self.nJeval = 0
        self.ndecomp = 0
        self.nreject = 0
        self.nsolve = 0
        self.ret = None
        self.succeed = False

    def __repr__(self):
        return (f"Scheme: {self.scheme}, "
                f"succeed: {self.succeed}, "
                f"nstep: {self.nstep}, "
                f"nfeval: {self.nfeval}, "
                f"nJeval: {self.nJeval}, "
                f"ndecomp: {self.ndecomp}, "
                f"nreject: {self.nreject}."
                f"nsolve: {self.nsolve}")
