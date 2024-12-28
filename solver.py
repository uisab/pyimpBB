'''This module contains methods for determining the global solution of nonlinear optimization problems.'''
import numpy as np
#import pickle

def improved_BandB(func, grad, cons, cons_div, X, bounding_procedure, epsilon = 0, delta = 0, epsilon_max = 0.0001, delta_max = 0.0001):
    """Uses the improvement function in the course of a branch-and-bound approach to provide an enclosure of the solution set with a given accuracy. 
    The arguments have to be a real objective function 'func' with associated gradient 'grad', a vector-valued constraint 'cons' with associated derivative 'cons_div', 
    a box 'X' surrounding the feasible set, a convergent bounding procedure 'bounding_procedure' to be used, an optimality accuracy 'epsilon', 
    a feasibility accuracy 'delta' and the respective enclosure accuracies 'epsilon_max' and 'delta_max'. The output corresponds to a three-tuple consisting of 
    a list of boxes 'O', whose union forms a superset of the solution set, the iteration number of the algorithm 'k' and the intermediate steps per iteration 'save'."""
    
    def bounding_omega(X,direct):
        return max((bounding_procedure(lambda x: cons(x)[i],lambda x: cons_div(x).T[i],X,direction=direct)[0] for i in range(len(cons(X)))))
        #return max((bounding_procedure(cons[i], cons_grad[i],X,direction=direct)[0] for i in range(len(cons(X)))))
        #return np.max(bounding_procedure(cons,cons_div,X,direction=direct)) #in Combi mit vectorwertigen boundings
    
    def bounding_f(X,direct):
        return bounding_procedure(func,grad,X,direction=direct)[0]
    
    def argmin_psi_e():
        lbs_omega = []
        lbs_f = []
        ub_f = bounding_f(X,"upper")
        min_lb_psi_e = np.inf
        Y_argmin_e = None
        for Y in L:
            lb_omega = bounding_omega(Y,"lower")
            lbs_omega.append(lb_omega)
            lb_f = bounding_f(Y,"lower")
            lbs_f.append(lb_f)
            lb_psi_e = max(lb_omega,lb_f -ub_f +epsilon)
            if lb_psi_e < min_lb_psi_e:
                min_lb_psi_e = lb_psi_e
                Y_argmin_e = Y
        return Y_argmin_e,lbs_omega,lbs_f,ub_f

    def argmin_psi_emax():
        min_lb_psi_emax = np.inf
        Y_argmin_emax = None
        for Y,lb_omega,lb_f in zip(L,lbs_omega,lbs_f):
            lb_psi_emax = max(lb_omega,lb_f -ub_f +epsilon_max)
            if lb_psi_emax < min_lb_psi_emax:
                min_lb_psi_emax = lb_psi_emax
                Y_argmin_emax = Y   
        return Y_argmin_emax, min_lb_psi_emax

    #info zusatz
    k = 0
    save = {0:([(X,-np.inf,np.inf)],[X])}
    #end

    O,L = [(X,-np.inf,np.inf)],[X]

    O_to_split = O #[Oi for Oi in O if (Oi[1] < 0 or Oi[2] > delta_max)]
    while any(O_to_split):
        X_to_split = O_to_split[0][0] #wähle X des ersten Elements in O_to_split
        X1_X2 = X_to_split.split()
        O.remove(O_to_split[0])

        for X in X1_X2:
            l_omega = bounding_omega(X,"lower")
            if not l_omega > delta:
                #ub_f = bounding_f(X,"upper")
                #Y_argmin_e = min(L, key= lambda Y: max(bounding_omega(Y,"lower"),bounding_f(Y,"lower") -ub_f +epsilon)) 
                Y_argmin_e,lbs_omega,lbs_f,ub_f = argmin_psi_e() #effiziensgewinn durch speichern von bounding for all Y in L
                
                y_mid = Y_argmin_e.midpoint()
                
                ub_psi_e = max(np.max(cons(y_mid)),func(y_mid)-bounding_f(X,"lower") +epsilon)
                #ub_psi_e = max(max(cons_i(y_mid) for cons_i in cons),func(y_mid)-bounding_f(X,"lower") +epsilon)
                
                if not ub_psi_e < 0:
                    #Y_argmin_emax = min(L, key= lambda Y: max(bounding_omega(Y,"lower"),bounding_f(Y,"lower") -ub_f +epsilon_max))
                    Y_argmin_emax, gamma_X = argmin_psi_emax() #**
                    
                    delta_X = bounding_omega(X,"upper")
                    O.append((X,gamma_X,delta_X))
                    
                    if Y_argmin_e == Y_argmin_emax: #try except else potentiell effizienter, wenn deutlich mehr != als == 
                        L.remove(Y_argmin_e)
                        L.extend(Y_argmin_e.split())
                    else:
                        L.remove(Y_argmin_e)
                        L.remove(Y_argmin_emax)
                        L.extend(Y_argmin_e.split())
                        L.extend(Y_argmin_emax.split())
        
        O_to_split = [Oi for Oi in O if (Oi[1] < 0 or Oi[2] > delta_max)]

        #info zusatz
        k += 1
        save[k] = (O.copy(),L.copy())

    #file = open(data_loc,"wb")
    #pickle.dump(save, file)
    #file.close()
    #end

    return O ,k ,save