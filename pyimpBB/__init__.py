# Copyright (c) 2024, Marc Rodestock <marc@fam-rodestock.de>
# All rights reserved.
# See LICENSE for details.

"""A branch-and-bound method using the improvement function in Python

This package provides the implementation of a novel branch-and-bound algorithm for the outer approximation 
of all global minimal points of a nonlinear constrained optimization problem using the improvement function, 
internally referred to as 'impfunc_BandB', to the corresponding publication 'The improvement function in 
branch-and-bound methods for complete global optimization' by S. Schwarze, O. Stein, P. Kirst and M. Rodestock."""

from interval import interval

def __width(self):
        """The interval consisting only of the width of each component."""
        return self.new(self.Component(x,x) for x in (c.sup - c.inf for c in self))

interval.width = property(__width)