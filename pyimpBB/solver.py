'''This module contains methods for determining the global solution of nonlinear optimization problems.'''
import numpy as np
#import pickle
from pyimpBB.helper import obvec,obmat,intvec
from typing import Callable, Tuple, List
from itertools import zip_longest

def improved_BandB(func: Callable[[obvec],float], cons: List[Callable[[obvec],float]], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                   grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, cons_grad: List[Callable[[obvec],obvec]]=[], cons_hess: List[Callable[[obvec],obmat]]=[], epsilon: float = 0, 
                   delta: float = 0, epsilon_max: float = 0.5, delta_max: float = 0.5, k_max: int = 2500) -> Tuple[list,int,dict]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear constrained optimization problem with a given accuracy. 
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative, 
    up to three python lists 'cons', 'cons_grad' and 'cons_hess' each containing python funktions, which correspond to the constraints with associated first and second derivatives, 
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    a float-value 'delta' as feasibility accuracy, the float-values 'epsilon_max' and 'delta_max' as respective enclosure accuracies and the integer 'k_max' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, 
    the iteration number of the algorithm 'k' and the intermediate steps per iteration 'save'."""
    
    def bounding_omega(X,direct):
        #return max((bounding_procedure(lambda x: cons(x)[i],lambda x: cons_jac(x).T[i],lambda x: cons_hessvec(x)[i],X,direction=direct)[0] for i in range(len(cons(X))))) #vektorwertige Funktion cons
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for cons_i,cons_grad_i,cons_hess_i in zip_longest(cons,cons_grad,cons_hess))) #liste reeller Funktionen cons_1, ..., cons_n
        #return np.max(bounding_procedure(cons, cons_div, cons_hessvec, X, direction=direct)) #in Combi mit vektorwertigen boundings
    
    lb_omega_Y = bounding_omega(X,"lower")
    lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
    #info zusatz
    k = 0
    save = {0:([(X,-np.inf,np.inf)],[(X,lb_omega_Y,lb_f_Y)])}
    #end
    
    O,L = [],[(X,lb_omega_Y,lb_f_Y)]

    O_to_split = [(X,-np.inf,np.inf)]
    while O_to_split and k < k_max:
        
        Oi = O_to_split[0] #Breitensuche alternativ auch mit O_next dann aber save je iteration nicht möglich
        del O_to_split[0]
        #Oi = O_to_split.pop() #Tiefensuche ohne for-Schleife  
        X1_X2 = Oi[0].split()

        for Xi in X1_X2:
            l_omega_X = bounding_omega(Xi,"lower")

            if not l_omega_X > delta:
                ub_f = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
                L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f +epsilon)) 
                
                y_mid = L_argmin_e[0].midpoint()
                
                #ub_psi_e = max(np.max(cons(y_mid)),func(y_mid)-bounding_procedure(func,grad,hess,Xi,direction="lower")[0] +epsilon)
                ub_psi_e = max(max(cons_i(y_mid) for cons_i in cons),func(y_mid)-bounding_procedure(func,grad,hess,Xi,direction="lower")[0] +epsilon)
                
                if not ub_psi_e < 0:
                    L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f +epsilon_max))
                    
                    gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f +epsilon_max)
                    delta_X = bounding_omega(Xi,"upper")
                    if gamma_X < 0 or delta_X > delta_max:
                        O_to_split.append((Xi,gamma_X,delta_X))
                    else:
                        O.append((Xi,gamma_X,delta_X))
                    
                    if id(L_argmin_e) == id(L_argmin_emax):
                        Y1,Y2 = L_argmin_e[0].split()
                        lb_omega_Y1 = bounding_omega(Y1,"lower")
                        lb_omega_Y2 = bounding_omega(Y2,"lower")
                        lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                        lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]
                        
                        L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                        L.extend([(Y1,lb_omega_Y1,lb_f_Y1),(Y2,lb_omega_Y2,lb_f_Y2)])
                    else:
                        Y1_e,Y2_e = L_argmin_e[0].split()
                        lb_omega_Y1_e = bounding_omega(Y1_e,"lower")
                        lb_omega_Y2_e = bounding_omega(Y2_e,"lower")
                        lb_f_Y1_e = bounding_procedure(func,grad,hess,Y1_e,direction="lower")[0]
                        lb_f_Y2_e = bounding_procedure(func,grad,hess,Y2_e,direction="lower")[0]

                        Y1_emax,Y2_emax = L_argmin_emax[0].split()
                        lb_omega_Y1_emax = bounding_omega(Y1_emax,"lower")
                        lb_omega_Y2_emax = bounding_omega(Y2_emax,"lower")
                        lb_f_Y1_emax = bounding_procedure(func,grad,hess,Y1_emax,direction="lower")[0]
                        lb_f_Y2_emax = bounding_procedure(func,grad,hess,Y2_emax,direction="lower")[0]

                        L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                        L.extend([(Y1_e,lb_omega_Y1_e,lb_f_Y1_e),(Y2_e,lb_omega_Y2_e,lb_f_Y2_e),(Y1_emax,lb_omega_Y1_emax,lb_f_Y1_emax),(Y2_emax,lb_omega_Y2_emax,lb_f_Y2_emax)])

        #info zusatz
        k += 1
        O_iter = O.copy()
        O_iter.extend(O_to_split)
        save[k] = (O_iter,L.copy())

    #file = open(data_loc,"wb")
    #pickle.dump(save, file)
    #file.close()
    #end

    return O ,k ,save

def improved_boxres_BandB(func: Callable[[obvec],float], X: intvec, bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], 
                          grad: Callable[[obvec],obvec]=None, hess: Callable[[obvec],obmat]=None, epsilon: float = 0, epsilon_max: float = 0.5, k_max: int = 2500) -> Tuple[list,int,dict]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set of a nonlinear box-constrained optimization problem with a given accuracy. 
    The arguments have to be up to three python functions 'func', 'grad' and 'hess', which correspond to the real objective function with associated first and second derivative,  
    a intvec 'X' bounding and/or surrounding the feasible set, a python function 'bounding_procedure' providing a convergent bounding procedure to be used, a float-value 'epsilon' as optimality accuracy, 
    the float-value 'epsilon_max' as respective enclosure accuracies and the integer 'k_max' for maximum number of iterations.
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, 
    the iteration number of the algorithm 'k' and the intermediate steps per iteration 'save'."""
    
    lb_f_Y = bounding_procedure(func,grad,hess,X,direction="lower")[0]
    
    #info zusatz
    k = 0
    save = {0:([(X,-np.inf,np.inf)],[(X,lb_f_Y)])}
    #end
    
    O,L = [],[(X,lb_f_Y)]

    O_to_split = [(X,-np.inf,np.inf)]
    while O_to_split and k < k_max:
        
        Oi = O_to_split[0] #Breitensuche alternativ auch mit O_next dann aber save je iteration nicht möglich
        del O_to_split[0]
        #Oi = O_to_split.pop() #Tiefensuche ohne for-Schleife  
        X1_X2 = Oi[0].split()

        for Xi in X1_X2:
            ub_f = bounding_procedure(func,grad,hess,Xi,direction="upper")[0]
            L_argmin_e = min(L, key= lambda Li: Li[1] -ub_f +epsilon)
            
            y_mid = L_argmin_e[0].midpoint()
            
            ub_psi_e = func(y_mid)-bounding_procedure(func,grad,hess,Xi,direction="lower")[0] +epsilon
            
            if not ub_psi_e < 0:
                L_argmin_emax = min(L, key= lambda Li: Li[1] -ub_f +epsilon_max)

                gamma_X = L_argmin_emax[1] -ub_f +epsilon_max
                if gamma_X < 0:
                    O_to_split.append((Xi,gamma_X))
                else:
                    O.append((Xi,gamma_X))
                
                if id(L_argmin_e) == id(L_argmin_emax):
                    Y1,Y2 = L_argmin_e[0].split()
                    lb_f_Y1 = bounding_procedure(func,grad,hess,Y1,direction="lower")[0]
                    lb_f_Y2 = bounding_procedure(func,grad,hess,Y2,direction="lower")[0]
                    
                    L = [Li for Li in L if id(Li) != id(L_argmin_e)]
                    L.extend([(Y1,lb_f_Y1),(Y2,lb_f_Y2)])
                else:
                    Y1_e,Y2_e = L_argmin_e[0].split()
                    lb_f_Y1_e = bounding_procedure(func,grad,hess,Y1_e,direction="lower")[0]
                    lb_f_Y2_e = bounding_procedure(func,grad,hess,Y2_e,direction="lower")[0]

                    Y1_emax,Y2_emax = L_argmin_emax[0].split()
                    lb_f_Y1_emax = bounding_procedure(func,grad,hess,Y1_emax,direction="lower")[0]
                    lb_f_Y2_emax = bounding_procedure(func,grad,hess,Y2_emax,direction="lower")[0]

                    L = [Li for Li in L if (id(Li) != id(L_argmin_e) and id(Li) != id(L_argmin_emax))]
                    L.extend([(Y1_e,lb_f_Y1_e),(Y2_e,lb_f_Y2_e),(Y1_emax,lb_f_Y1_emax),(Y2_emax,lb_f_Y2_emax)])

        #info zusatz
        k += 1
        O_iter = O.copy()
        O_iter.extend(O_to_split)
        save[k] = (O_iter,L.copy())

    #file = open(data_loc,"wb")
    #pickle.dump(save, file)
    #file.close()
    #end

    return O ,k ,save