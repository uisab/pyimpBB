from pyinterval import interval
import numpy as np
from helper import obvec, intvec, obmat
from typing import Callable

def optimal_centerd_forms(func: Callable[[obvec],float|obvec], grad: Callable[[obvec],obvec|obmat], X: intvec, direction: str="lower") -> obvec:
    """Uses optimal centered forms to return the upper or lower bounds 
    of the real potentially vector-valued function 'func' on a interval-box 'X' in the form of an object-vector.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or the first derivative of 'func', 
    and a string 'direction', which specifies the bound (upper or lower) to be determined."""
    def F(X,c):
        return func(c) + grad(X)@(X - c) #np.matmul(grad(X),(X-c), out=np.zeros(1,dtype=object))
    L = grad(X)
    c = [0]*len(X)
    if(direction == "lower"):
        for i in range(len(X)):
            if(L[i][-1].sup <= 0):
                c[i] = X[i][-1].sup
            elif(L[i][0].inf >= 0):
                c[i] = X[i][0].inf
            else:
                c[i] = (L[i][-1].sup*X[i][0].inf - L[i][0].inf*X[i][-1].sup)/(L[i][-1].sup - L[i][0].inf)
        bounds = F(X,obvec(c))
        if isinstance(bounds,interval):
            bounds = intvec([bounds])
        return intvec(bounds).inf
        #return np.array([f[0].inf for f in F(X,c)])
    elif(direction == "upper"):
        for i in range(len(X)):
            if(L[i][0].sup <= 0):
                c[i] = X[i][0].inf
            elif(L[i][0].inf >= 0):
                c[i] = X[i][0].sup
            else:
                c[i] = (L[i][0].inf*X[i][0].inf - L[i][-1].sup*X[i][-1].sup)/(L[i][0].inf - L[i][-1].sup)
        bounds = F(X,obvec(c))
        if isinstance(bounds,interval):
            bounds = intvec([bounds])
        return intvec(bounds).sup
        #return np.array([f[0].sup for f in F(X,c)])
    else:
        raise ValueError("direction "+str(direction)+" is not supported, try 'lower' or 'upper'")
    
def centerd_forms(func, grad, X, direction="lower"):
    """Uses centered forms to return the upper or lower bounds 
    of the real potentially vector-valued function 'func' on a interval-box 'X' in the form of an object-vector.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or the first derivative of 'func', 
    and a string 'direction', which specifies the bound (upper or lower) to be determined."""
    def F(X,c):
        return func(c) + grad(X)@(X - c) #np.matmul(grad(X),(X-c), out=np.zeros(1,dtype=object))
    bounds = F(X,X.midpoint())
    if isinstance(bounds,interval):
        bounds = intvec([bounds])
    if(direction == "lower"):
        return intvec(bounds).inf
        #return np.array([f[0].inf for f in F(X,Interval_Vector_Midpoint(X))])
    elif(direction == "upper"):
        return intvec(bounds).sup
        #return np.array([f[0].sup for f in F(X,Interval_Vector_Midpoint(X))])
    else:
        raise ValueError("direction "+str(direction)+" is not supported, try 'lower' or 'upper'")
