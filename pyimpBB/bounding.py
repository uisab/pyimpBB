'''This Modul contains convergent bounding procedures as python functions for use in all other models of this package and beyond.'''
#from interval import interval
import numpy as np
from scipy.optimize import minimize, Bounds
from pyimpBB.helper import obvec, intvec, obmat, interval
from typing import Callable, Union

def optimal_centerd_forms(func: Callable[[obvec], float], grad: Callable[[obvec],obvec], hess, X: intvec, direction: str="lower") -> obvec:
    """Uses optimal centered forms to return an upper or lower bound 
    of the real function 'func' on the interval-vector 'X' in the form of an object-vector.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or first derivative of 'func', 
    a unused placeholder 'hess' and a string 'direction', which specifies the bound (upper or lower) to be determined."""
    def F(X,c):
        return func(c) + grad(X)@(X - c) #np.matmul(grad(X),(X-c), out=np.zeros(1,dtype=object))
    L = intvec(grad(X))
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
    
def centerd_forms(func: Callable[[obvec],Union[float,obvec]], grad: Callable[[obvec],Union[obvec,obmat]], hess, X: intvec, direction: str="lower") -> obvec:
    """Uses centered forms to return an upper or lower bound 
    of the real potentially vector-valued function 'func' on the interval-vector 'X' in the form of an object-vector. 
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or first derivative of 'func', 
    a unused placeholder 'hess' and a string 'direction', which specifies the bound (upper or lower) to be determined."""
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
    
def aBB_relaxation(func: Callable[[obvec],float], grad: Callable[[obvec],obvec], hess: Callable[[obvec],obmat], X: intvec, direction: str="lower") -> obvec:
    """Uses konvex relaxation via aBB method to return an upper or lower bound 
    of the real function 'func' on the interval-vector 'X' in the form of an object-vector.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or the first derivative of 'func', 
    a python function 'hess', which corresponds to the hessian or the second derivative of 'func',
    and a string 'direction', which specifies the bound (upper or lower) to be determined."""
    if(direction == "lower"):
        h, hg, hh = func, grad, hess
        lam = 1
    elif(direction == "upper"):
        h = lambda x: -func(x)
        hg = lambda x: -grad(x)
        hh = lambda x: -hess(x)
        lam = -1
    else:
        raise ValueError("direction '"+str(direction)+"' is not supported, try 'lower' or 'upper'")
    A = hh(X)
    beta = min(interval(A[i][i])[0].inf - sum(max(abs(interval(A[j][i])[0].inf),abs(interval(A[j][i])[0].sup)) for j in range(len(A)) if j != i) for i in range(len(A)))
    alpha = max(0, -beta)
    h_alpha = lambda x: h(x) + (alpha/2)*(X.inf -obvec(x))@(X.sup -obvec(x))
    hg_alpha = lambda x: hg(x) + (alpha/2)*(-X.sup -X.inf +2*obvec(x))
    bounds = Bounds(list(X.inf),list(X.sup))
    res = minimize(h_alpha, X.midpoint(), method='SLSQP', jac=hg_alpha, bounds=bounds)
    lb = lam*res.fun if res.success else lam*-np.inf #func(res.x) + (alpha/2)*(X.inf -obvec(res.x))@(X.sup -obvec(res.x))
    return obvec([lb])

def direct_intervalarithmetic(func: Callable[[obvec],float], grad, hess, X: intvec, direction: str="lower") -> obvec:
    """Uses pur interval arithmetic to return an upper or lower bound 
    of the real function 'func' on the interval-vector 'X' in the form of an object-vector.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    two unused placeholder 'grad' as well as 'hess'and a string 'direction', which specifies the bound (upper or lower) to be determined."""
    bounds = func(X)
    if isinstance(bounds,interval):
        bounds = intvec([bounds])
    if(direction == "lower"):
        return intvec(bounds).inf
        #return np.array([f[0].inf for f in func(X)])
    elif(direction == "upper"):
        return intvec(bounds).sup
        #return np.array([f[0].sup for f in func(X)])
    else:
        raise ValueError("direction "+str(direction)+" is not supported, try 'lower' or 'upper'")
