'''This module contains methods for determining the global solution of nonlinear optimization problems.'''
import numpy as np
import time
import random
from pyimpBB.helper import obvec,obmat,intvec
from typing import Callable, Tuple, List, Union
from itertools import zip_longest

def improved_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
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
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess)))
    
    lb_omega_Y = bounding_omega(X,"lower")
    lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
    k = 0
    
    O,L = [],[(X,lb_omega_Y,lb_f_Y)]
    y_best, v_best = None, np.inf

    O_to_split = [(X,-np.inf,np.inf)]
    while O_to_split and k < max_iter:
        
        Oi = O_to_split[0] #Breitensuche
        del O_to_split[0]
        #Oi = O_to_split.pop() #Tiefensuche
        X1_X2 = Oi[0].split()

        for Xi in X1_X2:
            lb_omega_X = bounding_omega(Xi,"lower")

            if not lb_omega_X > delta:
                lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                
                if not (v_best -lb_f_X +epsilon) < 0:
                    ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                    L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon)) 
                    
                    y_mid = L_argmin_e[0].midpoint()
                    v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                    if max(omega_mid,v_mid-v_best) < 0:
                        y_best, v_best = y_mid, v_mid
                    
                    ub_psi_e = max(omega_mid,v_mid -lb_f_X +epsilon)
                    
                    if not ub_psi_e < 0:
                        L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon_max))
                        
                        gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f_X +epsilon_max)
                        delta_X = bounding_omega(Xi,"upper")
                        if gamma_X < 0 or delta_X > delta_max:
                            O_to_split.append((Xi,gamma_X,delta_X))
                        else:
                            O.append((Xi,gamma_X,delta_X))
                        
                        if id(L_argmin_e) == id(L_argmin_emax):
                            L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                            L_to_split = [L_argmin_e]
                        else:
                            L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                            L_to_split = [L_argmin_e,L_argmin_emax]
                        for L_i in L_to_split:
                            Y1,Y2 = L_i[0].split()
                            lb_omega_Y1 = bounding_omega(Y1,"lower")
                            lb_omega_Y2 = bounding_omega(Y2,"lower")
                            lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                            lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                            L.extend([(Y1,lb_omega_Y1,lb_f_Y1),(Y2,lb_omega_Y2,lb_f_Y2)])

        k += 1
    
    O.extend(O_to_split)

    return O ,y_best ,k

def improved_boxres_BandB(func: Callable[[obvec],float], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                          grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, epsilon: float = 0, epsilon_max: float = 0.5, max_iter: int = 2500) -> Tuple[list,obvec,int]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear box-constrained optimization problem with a given accuracy. 
    A modification of the more general main implementation for box-constrained problems or unconstrained problems with available search space restriction without any further additions.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative,  
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    the float-value 'epsilon_max' as respective enclosure accuracies and the integer 'max_iter' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best' and the iteration number of the algorithm 'k'."""
    
    lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
    k = 0
    
    O,L = [],[(X,lb_f_Y)]
    y_best, v_best = None, np.inf

    O_to_split = [(X,-np.inf)]
    while O_to_split and k < max_iter:
        
        Oi = O_to_split[0] #Breitensuche
        del O_to_split[0]
        #Oi = O_to_split.pop() #Tiefensuche
        X1_X2 = Oi[0].split()

        for Xi in X1_X2:
            lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                
            if not (v_best -lb_f_X +epsilon) < 0:
                ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                L_argmin_e = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon) 
                
                y_mid = L_argmin_e[0].midpoint()
                v_mid = func(y_mid)
                if v_mid-v_best < 0:
                    y_best, v_best = y_mid, v_mid
                
                ub_psi_e = v_mid -lb_f_X +epsilon
            
                if not ub_psi_e < 0:
                    L_argmin_emax = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon_max)

                    gamma_X = L_argmin_emax[1] -ub_f_X +epsilon_max
                    if gamma_X < 0:
                        O_to_split.append((Xi,gamma_X))
                    else:
                        O.append((Xi,gamma_X))
                    
                    if id(L_argmin_e) == id(L_argmin_emax):
                        L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                        L_to_split = [L_argmin_e]
                    else:
                        L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                        L_to_split = [L_argmin_e,L_argmin_emax]
                    for L_i in L_to_split:
                        Y1,Y2 = L_i[0].split()
                        lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                        lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                        L.extend([(Y1,lb_f_Y1),(Y2,lb_f_Y2)])

        k += 1
    
    O.extend(O_to_split)

    return O ,y_best ,k

def fathomed_improved_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                   grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, cons_grad: List[Callable[[obvec],obvec]]=[], cons_hess: List[Callable[[obvec],obmat]]=[], epsilon: float = 0, 
                   delta: float = 0, epsilon_max: float = 0.5, delta_max: float = 0.5, max_iter: int = 2500) -> Tuple[list,obvec,int]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear constrained optimization problem with a given accuracy. 
    A variant of the main implementation that fathomes the secondary list L to reduce the required memory and time per iteration as much as possible, thus handling more complex problems.
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative, 
    up to three python lists 'cons', 'cons_grad' and 'cons_hess' each containing python funktions, which correspond to the constraints with associated first and second derivatives, 
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    a float-value 'delta' as feasibility accuracy, the float-values 'epsilon_max' and 'delta_max' as respective enclosure accuracies and the integer 'max_iter' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, the carried best incumbent 'y_best' and the iteration number of the algorithm 'k'."""
    
    def bounding_omega(X,direct):
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess)))
    
    lb_omega_Y = bounding_omega(X,"lower")
    lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
    k = 0
    
    O,L = [],[(X,lb_omega_Y,lb_f_Y)]
    y_best, v_best = None, np.inf

    O_to_split = [(X,-np.inf,np.inf)]
    while O_to_split and k < max_iter:
        
        Oi = O_to_split[0] #Breitensuche
        del O_to_split[0]
        #Oi = O_to_split.pop() #Tiefensuche
        X1_X2 = Oi[0].split()

        for Xi in X1_X2:
            lb_omega_X = bounding_omega(Xi,"lower")

            if not lb_omega_X > delta:
                lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                
                if not (v_best -lb_f_X +epsilon) < 0:
                    ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                    L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon)) 
                    
                    y_mid = L_argmin_e[0].midpoint()
                    v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                    if max(omega_mid,v_mid-v_best) < 0:
                        y_best, v_best = y_mid, v_mid
                    
                    ub_psi_e = max(omega_mid,v_mid -lb_f_X +epsilon)
                    
                    if not ub_psi_e < 0:
                        L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon_max))
                        #L_argmin_emax = L_argmin_e if L_argmin_e[1] >= L_argmin_e[2]-ub_f_X +epsilon else min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon_max))
                        
                        gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f_X +epsilon_max)
                        delta_X = bounding_omega(Xi,"upper")
                        if gamma_X < 0 or delta_X > delta_max:
                            O_to_split.append((Xi,gamma_X,delta_X))
                        else:
                            O.append((Xi,gamma_X,delta_X))
                        
                        if id(L_argmin_e) == id(L_argmin_emax):
                            L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                            L_to_split = [L_argmin_e]
                        else:
                            L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                            L_to_split = [L_argmin_e,L_argmin_emax]
                        for L_i in L_to_split:
                            Y1_Y2 = L_i[0].split()
                            for Yi in Y1_Y2:
                                lb_omega_Yi = bounding_omega(Yi,"lower")
                                lb_f_Yi = bounding_procedure(func,grad,hess,Yi,direction="lower")[0]
                                lb_psi_e = max(lb_omega_Yi,lb_f_Yi -v_best -epsilon)
                                if not lb_psi_e > 0:
                                    L.append((Yi,lb_omega_Yi,lb_f_Yi))

        k += 1
        
    O.extend(O_to_split)

    return O ,y_best ,k

def analysed_improved_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
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
    the iteration number of the algorithm 'k', the required/elapsed time of the algorithm in seconds 't' and optionally an dict 'save' containing intermediate steps (O_k,L_k) per iteration 
    or a secondary list of boxes 'L' used for discard steps."""
    
    def bounding_omega(X,direct):
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess)))
    
    if save_lists:
        start = time.monotonic()

        lb_omega_Y = bounding_omega(X,"lower")
        lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
        
        k = 0
        save = {0:([(X,-np.inf,np.inf)],[(X,lb_omega_Y,lb_f_Y)])}

        O,L = [],[(X,lb_omega_Y,lb_f_Y)]
        y_best, v_best = None, np.inf

        O_to_split = [(X,-np.inf,np.inf)]
        while O_to_split and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Oi = O_to_split[bf_df]
            del O_to_split[bf_df]
            X1_X2 = Oi[0].split()

            for Xi in X1_X2:
                lb_omega_X = bounding_omega(Xi,"lower")

                if not lb_omega_X > delta:
                    lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                    
                    if not (v_best -lb_f_X +epsilon) < 0:
                        ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                        L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon)) 
                        
                        y_mid = L_argmin_e[0].midpoint()
                        v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                        if max(omega_mid,v_mid-v_best) < 0:
                            y_best, v_best = y_mid, v_mid
                        
                        ub_psi_e = max(omega_mid,v_mid -lb_f_X +epsilon)
                        
                        if not ub_psi_e < 0:
                            L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon_max))
                            
                            gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f_X +epsilon_max)
                            delta_X = bounding_omega(Xi,"upper")
                            if gamma_X < 0 or delta_X > delta_max:
                                O_to_split.append((Xi,gamma_X,delta_X))
                            else:
                                O.append((Xi,gamma_X,delta_X))
                            
                            if id(L_argmin_e) == id(L_argmin_emax):
                                L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                                L_to_split = [L_argmin_e]
                            else:
                                L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                                L_to_split = [L_argmin_e,L_argmin_emax]
                            for L_i in L_to_split:
                                Y1,Y2 = L_i[0].split()
                                lb_omega_Y1 = bounding_omega(Y1,"lower")
                                lb_omega_Y2 = bounding_omega(Y2,"lower")
                                lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                                lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                                L.extend([(Y1,lb_omega_Y1,lb_f_Y1),(Y2,lb_omega_Y2,lb_f_Y2)])

            k += 1
            O_iter = O.copy()
            O_iter.extend(O_to_split)
            save[k] = (O_iter,L.copy()) 
            
        t = time.monotonic() -start
        O.extend(O_to_split)

        return O, y_best, k, t, save

    else:
        start = time.monotonic()

        lb_omega_Y = bounding_omega(X,"lower")
        lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
        
        k = 0
        
        O,L = [],[(X,lb_omega_Y,lb_f_Y)]
        y_best, v_best = None, np.inf

        O_to_split = [(X,-np.inf,np.inf)]
        while O_to_split and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Oi = O_to_split[bf_df]
            del O_to_split[bf_df]
            X1_X2 = Oi[0].split()

            for Xi in X1_X2:
                lb_omega_X = bounding_omega(Xi,"lower")

                if not lb_omega_X > delta:
                    lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                    
                    if not (v_best -lb_f_X +epsilon) < 0:
                        ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                        L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon)) 
                        
                        y_mid = L_argmin_e[0].midpoint()
                        v_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
                        if max(omega_mid,v_mid-v_best) < 0:
                            y_best, v_best = y_mid, v_mid
                        
                        ub_psi_e = max(omega_mid,v_mid -lb_f_X +epsilon)
                        
                        if not ub_psi_e < 0:
                            L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f_X +epsilon_max))
                            
                            gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f_X +epsilon_max)
                            delta_X = bounding_omega(Xi,"upper")
                            if gamma_X < 0 or delta_X > delta_max:
                                O_to_split.append((Xi,gamma_X,delta_X))
                            else:
                                O.append((Xi,gamma_X,delta_X))
                            
                            if id(L_argmin_e) == id(L_argmin_emax):
                                L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                                L_to_split = [L_argmin_e]
                            else:
                                L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                                L_to_split = [L_argmin_e,L_argmin_emax]
                            for L_i in L_to_split:
                                Y1,Y2 = L_i[0].split()
                                lb_omega_Y1 = bounding_omega(Y1,"lower")
                                lb_omega_Y2 = bounding_omega(Y2,"lower")
                                lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                                lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                                L.extend([(Y1,lb_omega_Y1,lb_f_Y1),(Y2,lb_omega_Y2,lb_f_Y2)])

            k += 1
        
        t = time.monotonic() -start
        O.extend(O_to_split)

        return O, y_best, k, t, L
    
def analysed_improved_boxres_BandB(func: Callable[[obvec],float], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
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
    the iteration number of the algorithm 'k', the required/elapsed time of the algorithm in seconds 't' and optionally an dict 'save' containing intermediate steps (O_k,L_k) per iteration 
    or a secondary list of boxes 'L' used for discard steps."""
    
    if save_lists:
        start = time.monotonic()

        lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
        
        k = 0
        save = {0:([(X,-np.inf)],[(X,lb_f_Y)])}
        
        O,L = [],[(X,lb_f_Y)]
        y_best, v_best = None, np.inf

        O_to_split = [(X,-np.inf)]
        while O_to_split and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Oi = O_to_split[bf_df]
            del O_to_split[bf_df]
            X1_X2 = Oi[0].split()

            for Xi in X1_X2:
                lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                    
                if not (v_best -lb_f_X +epsilon) < 0:
                    ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                    L_argmin_e = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon) 
                    
                    y_mid = L_argmin_e[0].midpoint()
                    v_mid = func(y_mid)
                    if v_mid-v_best < 0:
                        y_best, v_best = y_mid, v_mid
                    
                    ub_psi_e = v_mid -lb_f_X +epsilon
                
                    if not ub_psi_e < 0:
                        L_argmin_emax = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon_max)

                        gamma_X = L_argmin_emax[1] -ub_f_X +epsilon_max
                        if gamma_X < 0:
                            O_to_split.append((Xi,gamma_X))
                        else:
                            O.append((Xi,gamma_X))
                        
                        if id(L_argmin_e) == id(L_argmin_emax):
                            L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                            L_to_split = [L_argmin_e]
                        else:
                            L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                            L_to_split = [L_argmin_e,L_argmin_emax]
                        for L_i in L_to_split:
                            Y1,Y2 = L_i[0].split()
                            lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                            lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                            L.extend([(Y1,lb_f_Y1),(Y2,lb_f_Y2)])

            k += 1
            O_iter = O.copy()
            O_iter.extend(O_to_split)
            save[k] = (O_iter,L.copy())
        
        t = time.monotonic() -start
        O.extend(O_to_split)

        return O, y_best, k, t, save

    else:
        start = time.monotonic()

        lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
        k = 0
        
        O,L = [],[(X,lb_f_Y)]
        y_best, v_best = None, np.inf

        O_to_split = [(X,-np.inf)]
        while O_to_split and (time.monotonic() -start) < max_time:
            
            bf_df = random.choices([-1,0],[search_ratio,1 -search_ratio])[0]
            Oi = O_to_split[bf_df]
            del O_to_split[bf_df]
            X1_X2 = Oi[0].split()

            for Xi in X1_X2:
                lb_f_X = bounding_procedure(func,grad,hess,Xi,direction="lower")[0]
                    
                if not (v_best -lb_f_X +epsilon) < 0:
                    ub_f_X = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                    L_argmin_e = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon) 
                    
                    y_mid = L_argmin_e[0].midpoint()
                    v_mid = func(y_mid)
                    if v_mid-v_best < 0:
                        y_best, v_best = y_mid, v_mid
                    
                    ub_psi_e = v_mid -lb_f_X +epsilon
                
                    if not ub_psi_e < 0:
                        L_argmin_emax = min(L, key= lambda Li: Li[1] -ub_f_X +epsilon_max)

                        gamma_X = L_argmin_emax[1] -ub_f_X +epsilon_max
                        if gamma_X < 0:
                            O_to_split.append((Xi,gamma_X))
                        else:
                            O.append((Xi,gamma_X))
                        
                        if id(L_argmin_e) == id(L_argmin_emax):
                            L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                            L_to_split = [L_argmin_e]
                        else:
                            L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                            L_to_split = [L_argmin_e,L_argmin_emax]
                        for L_i in L_to_split:
                            Y1,Y2 = L_i[0].split()
                            lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                            lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]

                            L.extend([(Y1,lb_f_Y1),(Y2,lb_f_Y2)])

            k += 1
        
        t = time.monotonic() -start
        O.extend(O_to_split)

        return O, y_best, k, t, L