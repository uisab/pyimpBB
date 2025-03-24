"""A branch-and-bound method using the improvement function in Python

This package provides the implementation of a novel branch-and-bound algorithm for the outer approximation 
of all global minimal points of a nonlinear constrained optimization problem using the improvement function, 
internally referred to as 'improved_BandB', to the corresponding publication 'The improvement function in 
branch-and-bound methods' by P. Kirst, M. Rodestock, S. Schwarze and O. Stein."""

from interval import interval

def __width(self):
        """The interval consisting only of the width of each component."""
        return self.new(self.Component(x,x) for x in (c.sup - c.inf for c in self))

interval.width = property(__width)