'''This module contains methods for determining all global solutions of nonlinear optimization problems.'''
import numpy as np
import time
from pyimpBB.helper import obvec,obmat,intvec
from typing import Callable, Tuple, List, Union
from itertools import zip_longest

def ICGO(func: Callable[[obvec], float], cons: List[Callable[[obvec], float]], X: intvec,
                            bounding_procedure: Callable[
                                [Callable[[obvec], float], Callable[[obvec], obvec], intvec, str], obvec],
                            grad: Callable[[obvec], obvec] = None, hess: Callable[[obvec], obmat] = None,
                            cons_grad: List[Callable[[obvec], obvec]] = [],
                            cons_hess: List[Callable[[obvec], obmat]] = [], epsilon: float = 0,
                            delta: float = 0, epsilon_max: float = 0.5, delta_max: float = 0.5, search_ratio: float = 0,
                            max_time: int = 60) -> Union[
    Tuple[list, obvec, int, float, dict], Tuple[list, obvec, int, float, list]]:

    def bounding_omega(X, direct):
        return max((bounding_procedure(cons_i, cons_grad_i, cons_hess_i, X, direction=direct)[0] for
                    cons_i, cons_grad_i, cons_hess_i in zip_longest(cons, cons_grad, cons_hess)))

    start = time.monotonic()

    lwx = bounding_omega(X, "lower")
    uwx = bounding_omega(X, "upper")
    lfx = bounding_procedure(func, grad, hess, X, direction="lower")[0]
    ufx = bounding_procedure(func, grad, hess, X, direction="upper")[0]

    k = 0

    O_init, O, W = [], [], [(X, lwx, uwx, lfx, ufx)]
    y_tilde, f_y_tilde = None, np.inf

    while W and (time.monotonic() - start) < max_time:
        ind = 0
        X_item = W[ind]
        X_discarded = False
        if X_item[1]>delta:
            X_discarded = True
            del W[ind]
        elif f_y_tilde - X_item[3] + epsilon < 0:
            X_discarded = True
            del W[ind]
        else:
            Y_1 = min(W, key=lambda Wi: max(Wi[1], Wi[3] - X_item[4] + epsilon))
            Y_hat_in_O = False
            if O:
                Y_2 = min(O, key=lambda Oi: max(Oi[1], Oi[3] - X_item[4] + epsilon))
                if max(Y_2[1], Y_2[3] - X_item[4] +epsilon)<max(Y_1[1], Y_1[3] - X_item[4] +epsilon):
                    Y_hat_in_O = True
                    Y_hat = Y_2
            if not Y_hat_in_O:
                Y_hat = Y_1
            y_mid = Y_hat[0].midpoint()
            f_mid, omega_mid = func(y_mid), max(cons_i(y_mid) for cons_i in cons)
            if max(omega_mid, f_mid - f_y_tilde) < 0:
                y_tilde, f_y_tilde = y_mid, f_mid
                if f_y_tilde - X_item[3] + epsilon < 0:
                    X_discarded = True
                    del W[ind]
        if not X_discarded:
            if not X_item[2] > delta_max:
                Y_1 = min(W, key=lambda Wi: max(Wi[1], Wi[3] - X_item[3] + epsilon_max))
                gamma_1 = max(Y_1[1], Y_1[3] - X_item[3] + epsilon_max)
                gamma_2 = +np.inf
                if O:
                    Y_2 = min(O, key=lambda Oi: max(Oi[1], Oi[3] - X_item[3] + epsilon_max))
                    gamma_2 = max(Y_2[1], Y_2[3] - X_item[3] + epsilon_max)
                gamma = min(gamma_1,gamma_2)
                if not gamma<0:
                    X_discarded = True
                    del W[ind]
                    O_init.append(X_item)
                    O.append(X_item)
            if not X_discarded:
                X1, X2 = X_item[0].split()
                del W[ind]
                lwx1 = bounding_omega(X1, "lower")
                lwx2 = bounding_omega(X2, "lower")
                uwx1 = bounding_omega(X1, "upper")
                uwx2 = bounding_omega(X2, "upper")
                lfx1 = bounding_procedure(func, grad, hess, X1, direction="lower")[0]
                lfx2 = bounding_procedure(func, grad, hess, X2, direction="lower")[0]
                ufx1 = bounding_procedure(func, grad, hess, X1, direction="upper")[0]
                ufx2 = bounding_procedure(func, grad, hess, X2, direction="lower")[0]
                W.extend([(X1, lwx1, uwx1, lfx1, ufx1),
                          (X2, lwx2, uwx2, lfx2, ufx2)])
                if X_item != Y_hat:
                    Y1, Y2 = Y_hat[0].split()
                    lwy1 = bounding_omega(Y1, "lower")
                    lwy2 = bounding_omega(Y2, "lower")
                    lfy1 = bounding_procedure(func, grad, hess, Y1, direction="lower")[0]
                    lfy2 = bounding_procedure(func, grad, hess, Y2, direction="lower")[0]
                    if Y_hat_in_O: #Note: We do not need upper bounds in the list O
                        O = [Oi for Oi in O if id(Oi) != id(Y_hat)]
                        O.extend([(Y1,lwy1,None,lfy1,None),
                                  (Y2,lwy2,None,lfy2,None)])
                    else:
                        W = [Wi for Wi in W if id(Wi) != id(Y_hat)]
                        uwy1 = bounding_omega(Y1, "upper")
                        uwy2 = bounding_omega(Y2, "upper")
                        ufy1 = bounding_procedure(func, grad, hess, Y1, direction="upper")[0]
                        ufy2 = bounding_procedure(func, grad, hess, Y2, direction="upper")[0]
                        W.extend([(Y1,lwy1,uwy1,lfy1,ufy1),
                                  (Y2,lwy2,uwy2,lfy2,ufy2)])

        k += 1
    t = time.monotonic() - start
    return O_init, y_tilde, k, t

