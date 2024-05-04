class Stats:

    def __init__(self, scheme=None):
        self.scheme = scheme
        self.nstep = 0
        self.nfeval = 0
        self.ndecomp = 0
        self.nreject = 0
        self.ret = None

    def __repr__(self):
        return f"Scheme {self.scheme}, nstep: {self.nstep}, nfeval: {self.nfeval}, ndecomp: {self.ndecomp}, nreject: {self.nreject}."
