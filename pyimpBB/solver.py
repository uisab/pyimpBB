'''This module contains methods for determining the global solution of nonlinear optimization problems.'''
import numpy as np
import time
import random
from pyimpBB.helper import obvec,obmat,intvec
from typing import Callable, Tuple, List, Union
from itertools import zip_longest, chain

def impfunc_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                   grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, cons_grad: List[Callable[[obvec],obvec]]=[], cons_hess: List[Callable[[obvec],obmat]]=[], epsilon: float = 0, 
                   delta: float = 0, epsilon_max: float = 0.5, delta_max: float = 0.5, max_iter: int = 2500) -> Tuple[list,obvec,int]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear constrained optimization problem with a given accuracy. 
    This is the main implementation in line with the publication of the corresponding approach provided with implemented breadth-first search, numerical emergency brake and without additional data collection.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative, 
    up to three python lists 'cons', 'cons_grad' and 'cons_hess' each containing python funktions, which correspond to the constraints with associated first and second derivatives, 
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    a float-value 'delta' as feasibility accuracy, the float-values 'epsilon_max' and 'delta_max' as respective enclosure accuracies and the integer 'max_iter' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best' and the iteration number of the algorithm 'k'."""
    
    def bounding_omega(X,direct):
        nonlocal cons, cons_grad, cons_hess
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess)))
    
    def subdivide_XY():
        nonlocal Pick, WO_argmin_e, W, O, from_O
        X1_X2 = Pick[0].split()
        for Xi in X1_X2:
            lb_omega_Xi = bounding_omega(Xi,"lower")
            lb_f_Xi = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
            W.append((Xi,lb_omega_Xi,lb_f_Xi))
        if id(Pick) != id(WO_argmin_e):
            Y1_Y2 = WO_argmin_e[0].split()
            if from_O:
                O = [Oi for Oi in O if id(Oi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_omega_Yi = bounding_omega(Yi,"lower")
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    O.append((Yi,lb_omega_Yi,lb_f_Yi))    
            else:
                W = [Wi for Wi in W if id(Wi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_omega_Yi = bounding_omega(Yi,"lower")
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    W.append((Yi,lb_omega_Yi,lb_f_Yi))
    
    lb_omega_X = bounding_omega(X,"lower")
    lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

    k = 0
    
    O_init, O, W = [], [], [(X,lb_omega_X,lb_f_X)]
    y_best, v_best = None, np.inf

    while W and k < max_iter:
        
        Pick = W[0] #Breitensuche
        del W[0]
        #Pick = W.pop() #Tiefensuche
        X,lb_omega_X,lb_f_X = Pick

        if not lb_omega_X > delta: #not subotimal
            
            if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
                ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
                W_argmin_e = min(chain([Pick],W), key= lambda Wi: max(Wi[1],Wi[2] -ub_f_X +epsilon))
                WO_argmin_e, from_O = W_argmin_e, False
                if O:
                    O_argmin_e = min(O, key= lambda Oi: max(Oi[1],Oi[2] -ub_f_X +epsilon))
                    if max(O_argmin_e[1], O_argmin_e[2] -ub_f_X +epsilon) < max(W_argmin_e[1], W_argmin_e[2] -ub_f_X +epsilon):
                        WO_argmin_e, from_O = O_argmin_e, True  
                    
                y_mid = WO_argmin_e[0].midpoint()
                v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                if max(omega_mid,v_mid -v_best) < 0:
                    y_best, v_best = y_mid, v_mid
                    
                    if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                        ub_omega_X = bounding_omega(X,"upper")

                        if not ub_omega_X > delta_max:
                            gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                            if not gamma_WO < 0: #tolerance fulfilling
                                O.append((X,lb_omega_X,lb_f_X))
                                O_init.append((X,lb_omega_X,lb_f_X))
                            else: #not tolerance fulfilling
                                subdivide_XY()
                        else: #not tolerance fulfilling
                            subdivide_XY()
                    else: #subotimal
                        pass
                else: #not subotimal
                    ub_omega_X = bounding_omega(X,"upper")

                    if not ub_omega_X > delta_max:
                        gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                        if not gamma_WO < 0: #tolerance fulfilling
                            O.append((X,lb_omega_X,lb_f_X))
                            O_init.append((X,lb_omega_X,lb_f_X))
                        else: #not tolerance fulfilling
                            subdivide_XY()
                    else: #not tolerance fulfilling
                        subdivide_XY()
            else: #subotimal
                pass
        else: #subotimal
            pass

        k += 1

    return O_init ,y_best ,k

def impfunc_boxres_BandB(func: Callable[[obvec],float], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                          grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, epsilon: float = 0, epsilon_max: float = 0.5, max_iter: int = 2500) -> Tuple[list,obvec,int]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear box-constrained optimization problem with a given accuracy. 
    A modification of the more general main implementation for box-constrained problems or unconstrained problems with available search space restriction without any further additions.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative,  
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    the float-value 'epsilon_max' as respective enclosure accuracies and the integer 'max_iter' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best' and the iteration number of the algorithm 'k'."""
    
    def subdivide_XY():
        nonlocal Pick, WO_argmin_e, W, O, from_O
        X1_X2 = Pick[0].split()
        for Xi in X1_X2:
            lb_f_Xi = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
            W.append((Xi,lb_f_Xi))
        if id(Pick) != id(WO_argmin_e):
            Y1_Y2 = WO_argmin_e[0].split()
            if from_O:
                O = [Oi for Oi in O if id(Oi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    O.append((Yi,lb_f_Yi))    
            else:
                W = [Wi for Wi in W if id(Wi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    W.append((Yi,lb_f_Yi))
    
    lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

    k = 0
    
    O_init, O, W = [], [], [(X,lb_f_X)]
    y_best, v_best = None, np.inf

    while W and k < max_iter:
        
        Pick = W[0] #Breitensuche
        del W[0]
        #Pick = W.pop() #Tiefensuche
        X,lb_f_X = Pick
            
        if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
            ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
            W_argmin_e = min(chain([Pick],W), key= lambda Wi: Wi[1] -ub_f_X +epsilon)
            WO_argmin_e, from_O = W_argmin_e, False
            if O:
                O_argmin_e = min(O, key= lambda Oi: Oi[1] -ub_f_X +epsilon)
                if (O_argmin_e[1] -ub_f_X +epsilon) < (W_argmin_e[1] -ub_f_X +epsilon):
                    WO_argmin_e, from_O = O_argmin_e, True  
                
            y_mid = WO_argmin_e[0].midpoint()
            v_mid = func(y_mid)
            if (v_mid -v_best) < 0:
                y_best, v_best = y_mid, v_mid
                
                if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                    gamma_WO = (WO_argmin_e[1] -ub_f_X +epsilon_max) #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                    
                    if not gamma_WO < 0: #tolerance fulfilling
                        O.append((X,lb_f_X))
                        O_init.append((X,lb_f_X))
                    else: #not tolerance fulfilling
                        subdivide_XY()
                else: #subotimal
                    pass
            else: #not subotimal
                gamma_WO = (WO_argmin_e[1] -ub_f_X +epsilon_max) #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                
                if not gamma_WO < 0: #tolerance fulfilling
                    O.append((X,lb_f_X))
                    O_init.append((X,lb_f_X))
                else: #not tolerance fulfilling
                    subdivide_XY()
        else: #subotimal
            pass

        k += 1

    return O_init ,y_best ,k

def analysed_impfunc_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                   grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, cons_grad: List[Callable[[obvec],obvec]]=[], cons_hess: List[Callable[[obvec],obmat]]=[], epsilon: float = 0, 
                   delta: float = 0, epsilon_max: float = 0.5, delta_max: float = 0.5, search_ratio: float = 0, max_time: int = 60, save_lists: bool = True) -> Union[Tuple[list,obvec,int,float,dict],Tuple[list,obvec,int,float,list]]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear constrained optimization problem with a given accuracy. 
    A variation of the main implementation that provides mixed breadth-depth-first search, a numerically useful second termination condition and collects additional data generally 
    and optionally per iteration to support subsequent analysis of the approximation progress and results.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative, 
    up to three python lists 'cons', 'cons_grad' and 'cons_hess' each containing python funktions, which correspond to the constraints with associated first and second derivatives, 
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, 
    a float-value 'epsilon' as optimality accuracy, a float-value 'delta' as feasibility accuracy, the float-values 'epsilon_max' and 'delta_max' as respective enclosure accuracies, 
    a float-value 'search_ratio' as the probability ratio of breadth-first to depth-first search (0 - 1 : bf - df search), an integer 'max_time' for the maximum runtime in seconds 
    and an optional flag for data collection per iteration 'save_lists'.
    The output corresponds to a five tuple, consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best', 
    the iteration number of the algorithm 'k', the required/elapsed time of the algorithm in seconds 't' and optionally an dict 'save' containing intermediate steps (O_k,W_k) per iteration 
    or a secondary to-do list of boxes 'W'."""

    def bounding_omega(X,direct):
            nonlocal cons, cons_grad, cons_hess
            return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess)))
        
    def subdivide_XY():
        nonlocal Pick, WO_argmin_e, W, O, from_O
        X1_X2 = Pick[0].split()
        for Xi in X1_X2:
            lb_omega_Xi = bounding_omega(Xi,"lower")
            lb_f_Xi = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
            W.append((Xi,lb_omega_Xi,lb_f_Xi))
        if id(Pick) != id(WO_argmin_e):
            Y1_Y2 = WO_argmin_e[0].split()
            if from_O:
                O = [Oi for Oi in O if id(Oi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_omega_Yi = bounding_omega(Yi,"lower")
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    O.append((Yi,lb_omega_Yi,lb_f_Yi))    
            else:
                W = [Wi for Wi in W if id(Wi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_omega_Yi = bounding_omega(Yi,"lower")
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    W.append((Yi,lb_omega_Yi,lb_f_Yi))

    if save_lists:
        start = time.monotonic()

        lb_omega_X = bounding_omega(X,"lower")
        lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

        k = 0
        save = {0:([],[(X,lb_omega_X,lb_f_X)])}
        
        O_init, O, W = [], [], [(X,lb_omega_X,lb_f_X)]
        y_best, v_best = None, np.inf

        while W and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Pick = W[bf_df]
            del W[bf_df]
            X,lb_omega_X,lb_f_X = Pick

            if not lb_omega_X > delta: #not subotimal
                
                if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
                    ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
                    W_argmin_e = min(chain([Pick],W), key= lambda Wi: max(Wi[1],Wi[2] -ub_f_X +epsilon))
                    WO_argmin_e, from_O = W_argmin_e, False
                    if O:
                        O_argmin_e = min(O, key= lambda Oi: max(Oi[1],Oi[2] -ub_f_X +epsilon))
                        if max(O_argmin_e[1], O_argmin_e[2] -ub_f_X +epsilon) < max(W_argmin_e[1], W_argmin_e[2] -ub_f_X +epsilon):
                            WO_argmin_e, from_O = O_argmin_e, True  
                        
                    y_mid = WO_argmin_e[0].midpoint()
                    v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                    if max(omega_mid,v_mid -v_best) < 0:
                        y_best, v_best = y_mid, v_mid
                        
                        if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                            ub_omega_X = bounding_omega(X,"upper")

                            if not ub_omega_X > delta_max:
                                gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                                if not gamma_WO < 0: #tolerance fulfilling
                                    O.append((X,lb_omega_X,lb_f_X))
                                    O_init.append((X,lb_omega_X,lb_f_X))
                                else: #not tolerance fulfilling
                                    subdivide_XY()
                            else: #not tolerance fulfilling
                                subdivide_XY()
                        else: #subotimal
                            pass
                    else: #not subotimal
                        ub_omega_X = bounding_omega(X,"upper")

                        if not ub_omega_X > delta_max:
                            gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                            if not gamma_WO < 0: #tolerance fulfilling
                                O.append((X,lb_omega_X,lb_f_X))
                                O_init.append((X,lb_omega_X,lb_f_X))
                            else: #not tolerance fulfilling
                                subdivide_XY()
                        else: #not tolerance fulfilling
                            subdivide_XY()
                else: #subotimal
                    pass
            else: #subotimal
                pass

            k += 1
            save[k] = (O_init.copy(),W.copy())
            
        t = time.monotonic() -start

        return O_init, y_best, k, t, save
    
    else:
        start = time.monotonic()

        lb_omega_X = bounding_omega(X,"lower")
        lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

        k = 0
        
        O_init, O, W = [], [], [(X,lb_omega_X,lb_f_X)]
        y_best, v_best = None, np.inf

        while W and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Pick = W[bf_df]
            del W[bf_df]
            X,lb_omega_X,lb_f_X = Pick

            if not lb_omega_X > delta: #not subotimal
                
                if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
                    ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
                    W_argmin_e = min(chain([Pick],W), key= lambda Wi: max(Wi[1],Wi[2] -ub_f_X +epsilon))
                    WO_argmin_e, from_O = W_argmin_e, False
                    if O:
                        O_argmin_e = min(O, key= lambda Oi: max(Oi[1],Oi[2] -ub_f_X +epsilon))
                        if max(O_argmin_e[1], O_argmin_e[2] -ub_f_X +epsilon) < max(W_argmin_e[1], W_argmin_e[2] -ub_f_X +epsilon):
                            WO_argmin_e, from_O = O_argmin_e, True  
                        
                    y_mid = WO_argmin_e[0].midpoint()
                    v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                    if max(omega_mid,v_mid -v_best) < 0:
                        y_best, v_best = y_mid, v_mid
                        
                        if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                            ub_omega_X = bounding_omega(X,"upper")

                            if not ub_omega_X > delta_max:
                                gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                                if not gamma_WO < 0: #tolerance fulfilling
                                    O.append((X,lb_omega_X,lb_f_X))
                                    O_init.append((X,lb_omega_X,lb_f_X))
                                else: #not tolerance fulfilling
                                    subdivide_XY()
                            else: #not tolerance fulfilling
                                subdivide_XY()
                        else: #subotimal
                            pass
                    else: #not subotimal
                        ub_omega_X = bounding_omega(X,"upper")

                        if not ub_omega_X > delta_max:
                            gamma_WO = min(max(WOi[1],WOi[2] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                            if not gamma_WO < 0: #tolerance fulfilling
                                O.append((X,lb_omega_X,lb_f_X))
                                O_init.append((X,lb_omega_X,lb_f_X))
                            else: #not tolerance fulfilling
                                subdivide_XY()
                        else: #not tolerance fulfilling
                            subdivide_XY()
                else: #subotimal
                    pass
            else: #subotimal
                pass

            k += 1

        t = time.monotonic() -start

        return O_init, y_best, k, t, W

def analysed_impfunc_boxres_BandB(func: Callable[[obvec],float], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                          grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, epsilon: float = 0, epsilon_max: float = 0.5, search_ratio: float = 0, 
                          max_time: int = 60, save_lists: bool = True) -> Union[Tuple[list,obvec,int,float,list],Tuple[list,obvec,int,float,dict]]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear box-constrained optimization problem with a given accuracy. 
    A variation of the modified implementation for box-constrained problems that provides mixed breadth-depth-first search, a numerically useful second termination condition and collects additional data generally 
    and optionally per iteration to support subsequent analysis of the approximation progress and results.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative,  
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    the float-value 'epsilon_max' as respective enclosure accuracies, a float-value 'search_ratio' as the probability ratio of breadth-first to depth-first search (0 - 1 : bf - df search), 
    an integer 'max_time' for the maximum runtime in seconds and an optional flag for data collection per iteration 'save_lists'.
    The output corresponds to a two-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best', 
    the iteration number of the algorithm 'k', the required/elapsed time of the algorithm in seconds 't' and optionally an dict 'save' containing intermediate steps (O_k,W_k) per iteration 
    or a secondary to-do list of boxes 'W'."""
        
    def subdivide_XY():
        nonlocal Pick, WO_argmin_e, W, O, from_O
        X1_X2 = Pick[0].split()
        for Xi in X1_X2:
            lb_f_Xi = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
            W.append((Xi,lb_f_Xi))
        if id(Pick) != id(WO_argmin_e):
            Y1_Y2 = WO_argmin_e[0].split()
            if from_O:
                O = [Oi for Oi in O if id(Oi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    O.append((Yi,lb_f_Yi))    
            else:
                W = [Wi for Wi in W if id(Wi) != id(WO_argmin_e)]
                for Yi in Y1_Y2:
                    lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                    W.append((Yi,lb_f_Yi))

    if save_lists:
        start = time.monotonic()

        lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

        k = 0
        save = {0:([],[(X,lb_f_X)])}
        
        O_init, O, W = [], [], [(X,lb_f_X)]
        y_best, v_best = None, np.inf

        while W and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Pick = W[bf_df]
            del W[bf_df]
            X,lb_f_X = Pick

            if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
                ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
                W_argmin_e = min(chain([Pick],W), key= lambda Wi: Wi[1] -ub_f_X +epsilon)
                WO_argmin_e, from_O = W_argmin_e, False
                if O:
                    O_argmin_e = min(O, key= lambda Oi: Oi[1] -ub_f_X +epsilon)
                    if (O_argmin_e[1] -ub_f_X +epsilon) < (W_argmin_e[1] -ub_f_X +epsilon):
                        WO_argmin_e, from_O = O_argmin_e, True  
                    
                y_mid = WO_argmin_e[0].midpoint()
                v_mid = func(y_mid)
                if (v_mid -v_best) < 0:
                    y_best, v_best = y_mid, v_mid
                    
                    if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                        gamma_WO = WO_argmin_e[1] -ub_f_X +epsilon_max #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                        
                        if not gamma_WO < 0: #tolerance fulfilling
                            O.append((X,lb_f_X))
                            O_init.append((X,lb_f_X))
                        else: #not tolerance fulfilling
                            subdivide_XY()
                    else: #subotimal
                        pass
                else: #not subotimal
                    gamma_WO = WO_argmin_e[1] -ub_f_X +epsilon_max #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))

                    if not gamma_WO < 0: #tolerance fulfilling
                        O.append((X,lb_f_X))
                        O_init.append((X,lb_f_X))
                    else: #not tolerance fulfilling
                        subdivide_XY()
            else: #subotimal
                pass

            k += 1
            save[k] = (O_init.copy(),W.copy())
            
        t = time.monotonic() -start

        return O_init, y_best, k, t, save
    
    else:
        start = time.monotonic()

        lb_f_X = bounding_procedure(func,grad,hess,X,direction="lower")[0]

        k = 0
        
        O_init, O, W = [], [], [(X,lb_f_X)]
        y_best, v_best = None, np.inf

        while W and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Pick = W[bf_df]
            del W[bf_df]
            X,lb_f_X = Pick

            if not (v_best -lb_f_X +epsilon) < 0: #not subotimal
                ub_f_X = bounding_procedure(func,grad,hess,X,direction="upper")[0]
                W_argmin_e = min(chain([Pick],W), key= lambda Wi: Wi[1] -ub_f_X +epsilon)
                WO_argmin_e, from_O = W_argmin_e, False
                if O:
                    O_argmin_e = min(O, key= lambda Oi: Oi[1] -ub_f_X +epsilon)
                    if (O_argmin_e[1] -ub_f_X +epsilon) < (W_argmin_e[1] -ub_f_X +epsilon):
                        WO_argmin_e, from_O = O_argmin_e, True  
                    
                y_mid = WO_argmin_e[0].midpoint()
                v_mid = func(y_mid)
                if (v_mid -v_best) < 0:
                    y_best, v_best = y_mid, v_mid
                    
                    if not (v_mid -lb_f_X +epsilon) < 0: #not subotimal
                        gamma_WO = WO_argmin_e[1] -ub_f_X +epsilon_max #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))
                        
                        if not gamma_WO < 0: #tolerance fulfilling
                            O.append((X,lb_f_X))
                            O_init.append((X,lb_f_X))
                        else: #not tolerance fulfilling
                            subdivide_XY()
                    else: #subotimal
                        pass
                else: #not subotimal
                    gamma_WO = WO_argmin_e[1] -ub_f_X +epsilon_max #min((WOi[1] -ub_f_X +epsilon_max) for WOi in chain([Pick],W,O))

                    if not gamma_WO < 0: #tolerance fulfilling
                        O.append((X,lb_f_X))
                        O_init.append((X,lb_f_X))
                    else: #not tolerance fulfilling
                        subdivide_XY()
            else: #subotimal
                pass

            k += 1
            
        t = time.monotonic() -start

        return O_init, y_best, k, t, W