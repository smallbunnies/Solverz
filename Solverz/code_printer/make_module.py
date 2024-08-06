from typing import List

from Solverz.code_printer.python import py_module_renderer
from Solverz.equation.equations import AE, FDAE, DAE
from Solverz.variable.variables import Vars


class module_printer:
    def __init__(self,
                 mdl: AE | FDAE | DAE,
                 variables: Vars | List[Vars],
                 name: str,
                 lang='python',
                 make_hvp=False,
                 directory=None,
                 jit=False):
        self.name = name
        self.lang = lang
        self.mdl = mdl
        if isinstance(variables, Vars):
            self.variables = [variables]
        else:
            self.variables = variables
        self.make_hvp = make_hvp
        self.directory = directory
        self.jit = jit

    def render(self):
        if self.lang == 'python':
            py_module_renderer(self.mdl,
                               *self.variables,
                               name=self.name,
                               directory=self.directory,
                               numba=self.jit,
                               make_hvp=self.make_hvp)
        else:
            raise NotImplemented(f"{self.lang} module renderer not implemented!")
