'''This Modul contains convergent bounding procedures as python functions for use in all other models of this package and beyond.'''
from interval import interval
import numpy as np
from scipy.optimize import minimize, Bounds
from pyimpBB.helper import obvec, intvec, obmat
from typing import Callable, Union

def optimal_centered_forms(func: Callable[[obvec], float], grad: Callable[[obvec],obvec], hess, X: intvec) -> intvec:
    """Uses optimal centered forms to return an upper and lower bound of the real potentially 
    vector-valued function 'func' on the interval-vector 'X' in the form of an enclosing interval.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or first derivative of 'func', 
    a unused placeholder 'hess' and a intvec 'X' as space restriction."""
    def F(X,c):
        return func(c) + grad(X)@(X - c) #np.matmul(grad(X),(X-c), out=np.zeros(1,dtype=object))
    L = intvec(grad(X))
    c_lb, c_ub = [0]*len(X), [0]*len(X)
    for i in range(len(X)):
        if(L[i][-1].sup <= 0):
            c_lb[i] = X[i][-1].sup
            c_ub[i] = X[i][0].inf
        elif(L[i][0].inf >= 0):
            c_lb[i] = X[i][0].inf
            c_ub[i] = X[i][-1].sup
        else:
            c_lb[i] = (L[i][-1].sup*X[i][0].inf - L[i][0].inf*X[i][-1].sup)/(L[i][-1].sup - L[i][0].inf)
            c_ub[i] = (L[i][0].inf*X[i][0].inf - L[i][-1].sup*X[i][-1].sup)/(L[i][0].inf - L[i][-1].sup)
    lower_bounds, upper_bounds = F(X,obvec(c_lb)), F(X,obvec(c_ub))
    if isinstance(lower_bounds,interval):
        lower_bounds = [lower_bounds]
    if isinstance(upper_bounds,interval):
        upper_bounds = [upper_bounds]
    return intvec(obmat([intvec(lower_bounds).inf,intvec(upper_bounds).sup]).T)
    
def centered_forms(func: Callable[[obvec],Union[float,obvec]], grad: Callable[[obvec],Union[obvec,obmat]], hess, X: intvec) -> intvec:
    """Uses centered forms to return an upper and lower bound of the real potentially 
    vector-valued function 'func' on the interval-vector 'X' in the form of an enclosing interval.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or first derivative of 'func', 
    a unused placeholder 'hess' and a intvec 'X' as space restriction."""
    def F(X,c):
        return func(c) + grad(X)@(X - c) #np.matmul(grad(X),(X-c), out=np.zeros(1,dtype=object))
    bounds = F(X,X.midpoint())
    if isinstance(bounds,interval):
        bounds = [bounds]
    return intvec(bounds)
    
def aBB_relaxation(func: Callable[[obvec],float], grad: Callable[[obvec],obvec], hess: Callable[[obvec],obmat], X: intvec) -> intvec:
    """Uses konvex relaxation via aBB method to return an upper and lower bound 
    of the real function 'func' on the interval-vector 'X' in the form of an enclosing interval.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    a python function 'grad', which corresponds to the gradient or the first derivative of 'func', 
    a python function 'hess', which corresponds to the hessian or the second derivative of 'func',
    and a intvec 'X' as space restriction."""
    A = hess(X)
    beta_lb = min(interval(A[i][i])[0].inf - sum(max(abs(interval(A[j][i])[0].inf),abs(interval(A[j][i])[0].sup)) for j in range(len(A)) if j != i) for i in range(len(A)))
    alpha_lb = max(0, -beta_lb)
    beta_ub = max(interval(A[i][i])[-1].sup + sum(max(abs(interval(A[j][i])[0].inf),abs(interval(A[j][i])[-1].sup)) for j in range(len(A)) if j != i) for i in range(len(A)))
    alpha_ub = min(0, -beta_ub)

    func_alpha_lb = lambda x: func(x) + (alpha_lb/2)*(X.inf -obvec(x))@(X.sup -obvec(x))
    func_alpha_lb_grad = lambda x: grad(x) + (alpha_lb/2)*(-X.sup -X.inf +2*obvec(x))
    func_alpha_ub = lambda x: -func(x) - (alpha_ub/2)*(X.inf -obvec(x))@(X.sup -obvec(x))
    func_alpha_ub_grad = lambda x: -grad(x) - (alpha_ub/2)*(-X.sup -X.inf +2*obvec(x))

    bounds = Bounds(list(X.inf),list(X.sup))
    res_lb = minimize(func_alpha_lb, X.midpoint(), method='SLSQP', jac=func_alpha_lb_grad, options={'ftol':1e-9}, bounds=bounds)
    lb = (res_lb.fun -1e-9) if res_lb.success else -np.inf #func(res_lb.x) + (alpha/2)*(X.inf -obvec(res_lb.x))@(X.sup -obvec(res_lb.x))
    res_ub = minimize(func_alpha_ub, X.midpoint(), method='SLSQP', jac=func_alpha_ub_grad, options={'ftol':1e-9}, bounds=bounds)
    ub = -(res_ub.fun -1e-9) if res_ub.success else np.inf #func(res_ub.x) + (alpha/2)*(X.inf -obvec(res_ub.x))@(X.sup -obvec(res_ub.x))
    return intvec([[lb,ub]])

def direct_intervalarithmetic(func: Callable[[obvec],float], grad, hess, X: intvec) -> intvec:
    """Uses pur interval arithmetic to return an upper and lower bound of the real potentially 
    vector-valued function 'func' on the interval-vector 'X' in the form of an enclosing interval.
    The arguments have to be a python function 'func', whose bounds are to be determined, 
    two unused placeholder 'grad' as well as 'hess' and a intvec 'X' as space restriction."""
    bounds = func(X)
    if isinstance(bounds,interval):
        bounds = [bounds]
    return intvec(bounds)
