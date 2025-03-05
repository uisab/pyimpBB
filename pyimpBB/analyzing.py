'''This module contains analyzing and mapping functions for application to the solvers included in this package.'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from interval import fpu
from pyimpBB.bounding import centerd_forms
from pyimpBB.helper import obvec,intvec
from typing import Callable, Union, List, Tuple

def iterations_in_decision_space_plot(func: Callable[[obvec],float],X: intvec,data: dict,iterations: list,cons: Union[List[Callable[[obvec],float]],None]=None,title: str="Iterations in decision space",
                                      fname: Union[str,None]=None,columns: int=3,levels: Union[int,List[float],None]=None,mgres:int = 100,xylim: Union[List[Tuple[float,float]],None]=None, **args) -> None:
    """Generates a tabular representation in which the decision space is shown for given iterations, 
    including level lines of the objective function, zero level lines of the constraints, enclosing box X and associated approximation or decomposition progress.
    The arguments have to be a python function 'func', which correspond to the real objective function, a intvec 'X' bounding and/or surrounding the feasible set, 
    a dictionary 'data' containing the corresponding box progress for a given iteration, a list of iterations to display, a optional list of python functions 'cons', which correspond to the constraints, 
    an optional title of the plot, an optional file name to save the plot, an optional number of columns, an optional numper or list of specified levels for the contour plot of 'func', 
    an optional value 'mgres' for the meshgrid resolution, an optional list of 'xy' dimension limits 'xylim' and additional optional parameters to 'plt.figure'."""

    rows = -(-len(iterations)//columns)

    if not xylim:
        xylim = [(X[0][0].inf -0.5,X[0][0].sup +0.5),(X[1][0].inf -0.5,X[1][0].sup +0.5)]
    
    X_1, X_2 = np.meshgrid(np.linspace(xylim[0][0],xylim[0][1],round(xylim[0][1] - xylim[0][0])*mgres),np.linspace(xylim[1][0],xylim[1][1],round(xylim[1][1] - xylim[1][0])*mgres))
    if cons:
        cons_eval = np.array([cons_i((X_1,X_2)) for cons_i in cons])
        fis_test = np.ones(shape=X_1.shape).astype(bool)
        for cons_bool in (cons_eval <= 0):
            fis_test = fis_test & cons_bool

    fig = plt.figure(**args,layout="constrained")
    fig.suptitle(title,fontsize="x-large",verticalalignment='center')
    fig.supxlabel(" ")
    
    for u,k in enumerate(iterations):
        ax = fig.add_subplot(rows,columns,u+1)

        ax.fill([X[0][0].inf,X[0][0].sup,X[0][0].sup,X[0][0].inf],[X[1][0].inf,X[1][0].inf,X[1][0].sup,X[1][0].sup],alpha=0.5, color="lightblue")

        if cons:
            for Z in cons_eval:
                ax.contour(X_1,X_2,Z,levels=[0],colors="purple")

            ax.imshow(fis_test.astype(int), extent=(xylim[0][0],xylim[0][1],xylim[1][0],xylim[1][1]),origin='lower',cmap='Purples',alpha=0.5,aspect='auto')

        if levels:
            CF = ax.contour(X_1,X_2,func((X_1,X_2)),levels,colors="darkred")
            ax.clabel(CF, CF.levels, fontsize=10)
        else:
            CF = ax.contour(X_1,X_2,func((X_1,X_2)),cmap="Reds")

        for B in data[k]:
            x_1 = [B[0][0].inf,B[0][0].sup,B[0][0].sup,B[0][0].inf]
            x_2 = [B[1][0].inf,B[1][0].inf,B[1][0].sup,B[1][0].sup]
            ax.fill(x_1,x_2,alpha=0.5,color="darkorange")

        ax.set_xlim(xylim[0])
        ax.set_ylim(xylim[1])
        ax.plot(xylim[0],[0,0],color="black")
        ax.plot([0,0],xylim[1],color="black")
        
        ax.grid(color="lightgray")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("View of the decision space in Iteration "+str(k))

    f_patch = mpatch.Patch(color="darkred", label="Objectiv function level lines")
    c_patch = mpatch.Patch(color="purple", label="Constraints zero level lines")
    X_patch = mpatch.Patch(color="lightblue", label="Enclosing box $X$")
    B_patch = mpatch.Patch(color="darkorange", label="Box-data")
    fig.legend(handles=[f_patch,c_patch,X_patch,B_patch], loc="lower center", ncol=4, mode="expand")
    
    if fname:
        plt.savefig(fname,bbox_inches='tight')
    plt.show()

#Creation and registration of a custom colormap for 'matplotlib' with a color gradient from transparent to purple
ncolors = 256
color_array = plt.get_cmap('Purples')(range(ncolors))
color_array[:,-1] = np.linspace(0.0,0.75,ncolors)
map_object = LinearSegmentedColormap.from_list(name='purple_alpha',colors=color_array)
plt.colormaps.register(cmap=map_object)

def iterations_in_objective_space_plot(func: Callable[[obvec],float],X: intvec,data: dict,iterations: list,grad: Union[Callable[[obvec],obvec],None]=None,cons: Union[List[Callable[[obvec],float]],None]=None,
                                       title: str='Iterations in objective space',fname: Union[str,None]=None,columns: int=3,dspace: bool=True,mgres:int = 100,xyzlim: Union[List[Tuple[float,float]],None]=None, **args) -> None:
    """Generates a tabular representation in which the objective space is shown for given iterations, 
    including the surface of the objective function, the associated optimal value approximation progress and optionally the decision space.
    The arguments have to be up to two python functions 'func' and 'grad',which correspond to the real objective function with associated gradient, a intvec 'X' bounding and/or surrounding the feasible set, 
    a dictionary 'data' containing the corresponding box progress for a given iteration, a list of iterations to display, a optional list of python functions 'cons', which correspond to the constraints, 
    an optional title of the plot, an optional file name to save the plot, an optional number of columns, an optional flag for plotting the decision space 'dspace', 
    an optional value 'mgres' for the meshgrid resolution, an optional list of 'xyz' dimension limits 'xyzlim' and additional optional parameters to 'plt.figure'."""
    
    rows = -(-len(iterations)//columns)
    if not xyzlim:
        if grad:
            lb_f, ub_f = centerd_forms(func,grad,None,X,direction="lower")[0], centerd_forms(func,grad,None,X,direction="upper")[0]
        else:
            lb_f, ub_f = -fpu.infinity, fpu.infinity
        if lb_f == -fpu.infinity or ub_f == fpu.infinity:
            testvalues = [X.midpoint(),obvec([X[0][0].inf,X[1][0].inf]),obvec([X[0][0].sup,X[1][0].inf]),obvec([X[0][0].sup,X[1][0].sup]),obvec([X[0][0].inf,X[1][0].sup])]
            lb_f, ub_f = min(func(tv) for tv in testvalues), max(func(tv) for tv in testvalues) #func(X.midpoint())-X.width(),func(X.midpoint)+X.width()
        if dspace and (lb_f > 0):
            lb_f = 0
        elif dspace and (ub_f < 0):
            ub_f = 0
        xyzlim = [(X[0][0].inf -0.5,X[0][0].sup +0.5),(X[1][0].inf -0.5,X[1][0].sup +0.5),(lb_f,ub_f)]

    X_1, X_2 = np.meshgrid(np.linspace(xyzlim[0][0],xyzlim[0][1],round(xyzlim[0][1] - xyzlim[0][0])*mgres),np.linspace(xyzlim[1][0],xyzlim[1][1],round(xyzlim[1][1] - xyzlim[1][0])*mgres))
    if cons:
        cons_eval = np.array([cons_i((X_1,X_2)) for cons_i in cons])
        fis_test = np.ones(shape=X_1.shape).astype(bool)
        for cons_bool in (cons_eval <= 0):
            fis_test = fis_test & cons_bool

    fig = plt.figure(**args,layout="constrained")
    fig.suptitle(title,fontsize="x-large")
    fig.supxlabel(" ")

    for u,k in enumerate(iterations):

        ax = fig.add_subplot(rows,columns,u+1,projection='3d')
        
        if dspace:
            Box_X = mpatch.Rectangle((X[0][0].inf,X[1][0].inf), X[0].width[0][0] ,X[1].width[0][0],alpha=0.5)
            ax.add_patch(Box_X)
            art3d.pathpatch_2d_to_3d(Box_X, z=0, zdir="z")

            if cons:
                for Z in cons_eval:
                    ax.contour(X_1,X_2,Z,levels=[0],colors="purple")

                ax.contourf(X_1,X_2,fis_test.astype(int),1, offset=0, cmap="purple_alpha")
            
            for B in data[k]:
                Box_B = mpatch.Rectangle((B[0][0].inf,B[1][0].inf), B[0].width[0][0] ,B[1].width[0][0],alpha=0.5,color="royalblue")
                ax.add_patch(Box_B)
                art3d.pathpatch_2d_to_3d(Box_B, z=0, zdir="z")
        
        func_eval = func((X_1,X_2))
        func_eval[(func_eval < xyzlim[2][0]) | (func_eval > xyzlim[2][1])] = np.nan
        ax.plot_surface(X_1, X_2, func_eval, cmap="Reds",alpha=0.5)

        for B in data[k]:
            Box_B_h = mpatch.Rectangle((B[0][0].inf,B[1][0].inf), B[0].width[0][0] ,B[1].width[0][0],alpha=0.8,color="darkorange")
            ax.add_patch(Box_B_h)
            art3d.pathpatch_2d_to_3d(Box_B_h, z=func(B.midpoint()), zdir="z")

        ax.set_xlim(xyzlim[0]) 
        ax.set_ylim(xyzlim[1])
        ax.set_zlim(xyzlim[2])
        ax.plot(xyzlim[0],[0,0],[0,0],color="black")
        ax.plot([0,0],xyzlim[1],[0,0],color="black")
        ax.plot([0,0],[0,0],xyzlim[2],color="black")
        ax.grid(color="lightgray")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$z$")
        ax.set_title("View of the object space in Iteration "+str(k))
    
    f_patch = mpatch.Patch(color="darkred", label="Objectiv function")
    c_patch = mpatch.Patch(color="purple", label="Constraints zero level lines")
    X_patch = mpatch.Patch(color="lightblue", label="Enclosing box $X$")
    B_patch = mpatch.Patch(color="darkorange", label="Box-data")
    fig.legend(handles=[f_patch,c_patch,X_patch,B_patch], loc="lower center", ncol=4, mode="expand")
    
    if fname:
        plt.savefig(fname,bbox_inches='tight')
    plt.show()