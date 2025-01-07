'''This module contains methods for determining the global solution of nonlinear optimization problems.'''
import numpy as np
#import pickle
from helper import obvec,obmat,intvec
from typing import Callable, Tuple

def improved_BandB(func: Callable[[obvec],float], grad: Callable[[obvec],obvec], cons: Callable[[obvec],obvec], cons_div: Callable[[obvec],obmat], X: intvec, 
                   bounding_procedure: Callable[[Callable[[obvec], float], Callable[[obvec],obvec], intvec, str],obvec], epsilon: float = 0, delta: float = 0, 
                   epsilon_max: float = 0.5, delta_max: float = 0.5, k_max: int = 2500) -> Tuple[list,int,dict]:
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set with a given accuracy. 
    The arguments have to be a real objective function 'func' with associated gradient 'grad', a vector-valued constraint 'cons' with associated derivative 'cons_div', 
    a box 'X' surrounding the feasible set, a convergent bounding procedure 'bounding_procedure' to be used, an optimality accuracy 'epsilon', 
    a feasibility accuracy 'delta', the respective enclosure accuracies 'epsilon_max' and 'delta_max' and the maximum number of iterations 'k_max'. 
    The output corresponds to a three-tuple consisting of a list of boxes 'O', whose union forms a superset of the solution set, 
    the iteration number of the algorithm 'k' and the intermediate steps per iteration 'save'."""
    
    def bounding_omega(X,direct):
        return max((bounding_procedure(lambda x: cons(x)[i],lambda x: cons_div(x).T[i],X,direction=direct)[0] for i in range(len(cons(X)))))
        #return max((bounding_procedure(cons[i], cons_grad[i],X,direction=direct)[0] for i in range(len(cons(X)))))
        #return np.max(bounding_procedure(cons,cons_div,X,direction=direct)) #in Combi mit vectorwertigen boundings
    
    lb_omega_Y = bounding_omega(X,"lower")
    lb_f_Y = bounding_procedure(func,grad,X,direction="lower")[0]
    
    #info zusatz
    k = 0
    save = {0:([(X,-np.inf,np.inf)],[(X,lb_omega_Y,lb_f_Y)])}
    #end
    
    O,L = [(X,-np.inf,np.inf)],[(X,lb_omega_Y,lb_f_Y)]

    O_to_split = O #[Oi for Oi in O if (Oi[1] < 0 or Oi[2] > delta_max)]
    while any(O_to_split) and k < k_max:
        X1_X2 = O_to_split[0][0].split() #wähle X des ersten Elements in O_to_split
        O.remove(O_to_split[0])

        for Xi in X1_X2:
            l_omega_X = bounding_omega(Xi,"lower")

            if not l_omega_X > delta:
                ub_f = bounding_procedure(func,grad,Xi,direction="upper")[0]
                L_argmin_e = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f +epsilon)) 
                
                y_mid = L_argmin_e[0].midpoint()
                
                ub_psi_e = max(np.max(cons(y_mid)),func(y_mid)-bounding_procedure(func,grad,Xi,direction="lower")[0] +epsilon)
                #ub_psi_e = max(max(cons_i(y_mid) for cons_i in cons),func(y_mid)-bounding_f(Xi,"lower") +epsilon)
                
                if not ub_psi_e < 0:
                    L_argmin_emax = min(L, key= lambda Li: max(Li[1],Li[2] -ub_f +epsilon_max))
                    
                    gamma_X = max(L_argmin_emax[1],L_argmin_emax[2] -ub_f +epsilon_max)
                    delta_X = bounding_omega(Xi,"upper")
                    O.append((Xi,gamma_X,delta_X))
                    
                    if id(L_argmin_e) == id(L_argmin_emax):
                        Y1,Y2 = L_argmin_e[0].split()
                        lb_omega_Y1 = bounding_omega(Y1,"lower")
                        lb_omega_Y2 = bounding_omega(Y2,"lower")
                        lb_f_Y1 = bounding_procedure(func,grad,Y1,direction="lower")[0]
                        lb_f_Y2 = bounding_procedure(func,grad,Y2,direction="lower")[0]
                        
                        L.remove(L_argmin_e)
                        L.extend([(Y1,lb_omega_Y1,lb_f_Y1),(Y2,lb_omega_Y2,lb_f_Y2)])
                    else:
                        Y1_e,Y2_e = L_argmin_e[0].split()
                        lb_omega_Y1_e = bounding_omega(Y1_e,"lower")
                        lb_omega_Y2_e = bounding_omega(Y2_e,"lower")
                        lb_f_Y1_e = bounding_procedure(func,grad,Y1_e,direction="lower")[0]
                        lb_f_Y2_e = bounding_procedure(func,grad,Y2_e,direction="lower")[0]

                        Y1_emax,Y2_emax = L_argmin_emax[0].split()
                        lb_omega_Y1_emax = bounding_omega(Y1_emax,"lower")
                        lb_omega_Y2_emax = bounding_omega(Y2_emax,"lower")
                        lb_f_Y1_emax = bounding_procedure(func,grad,Y1_emax,direction="lower")[0]
                        lb_f_Y2_emax = bounding_procedure(func,grad,Y2_emax,direction="lower")[0]

                        L.remove(L_argmin_e)
                        L.extend([(Y1_e,lb_omega_Y1_e,lb_f_Y1_e),(Y2_e,lb_omega_Y2_e,lb_f_Y2_e)])

                        L.remove(L_argmin_emax)
                        L.extend([(Y1_emax,lb_omega_Y1_emax,lb_f_Y1_emax),(Y2_emax,lb_omega_Y2_emax,lb_f_Y2_emax)])
        
        O_to_split = [Oi for Oi in O if (Oi[1] < 0 or Oi[2] > delta_max)]

        #info zusatz
        k += 1
        save[k] = (O.copy(),L.copy())

    #file = open(data_loc,"wb")
    #pickle.dump(save, file)
    #file.close()
    #end

    return O ,k ,save