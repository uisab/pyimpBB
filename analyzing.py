'''This module contains analyzing and mapping functions for application to the solvers included in this package.'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from bounding import centerd_forms
from helper import obvec,intvec
from typing import Callable, Union

def iterations_in_decision_space_plot(func: Callable[[obvec],float],cons: Union[Callable[[obvec],obvec],None],X: intvec,data: dict,iterations: list,columns: int=3,
                                      title: str="Iterations in decision space", fname: Union[str,None]=None, **args) -> None:
    """Generates a tabular representation in which the decision space is shown for given iterations, 
    including level lines of the objective function, zero level lines of the constraints, enclosing box X and associated approximation or decomposition progress.
    The arguments have to be a real objective function 'func', a vector-valued constraint 'cons' (or Python-value: None), 
    a box 'X' surrounding the feasible set, a dictionary containing the corresponding box progress for a given iteration, 
    a list of iterations to display, an optional number of columns, an optional title of the plot, an optional file name to save the plot 
    and additional optional parameters to 'plt.figure'."""

    rows = -(-len(iterations)//columns)

    fig = plt.figure(**args,layout="constrained")
    fig.suptitle(title,fontsize="x-large",verticalalignment='center')
    fig.supxlabel(" ")

    for u,k in enumerate(iterations):
        ax = fig.add_subplot(rows,columns,u+1)

        ax.fill([X[0][0].inf,X[0][0].sup,X[0][0].sup,X[0][0].inf],[X[1][0].inf,X[1][0].inf,X[1][0].sup,X[1][0].sup],alpha=0.5, color="lightblue")

        X_1, X_2 = np.meshgrid(np.linspace(X[0][0].inf -0.5,X[0][0].sup+0.5,round(X[0][0].sup - X[0][0].inf +1)*100),np.linspace(X[1][0].inf -0.5,X[1][0].sup +0.5,round(X[1][0].sup - X[1][0].inf +1)*100))
        
        if callable(cons):
            for Z in cons((X_1,X_2)):
                ax.contour(X_1,X_2,Z,levels=[0],colors="purple")

        ax.contour(X_1,X_2,func((X_1,X_2)),cmap="Reds")

        for B in data[k]:
            x_1 = [B[0][0].inf,B[0][0].sup,B[0][0].sup,B[0][0].inf]
            x_2 = [B[1][0].inf,B[1][0].inf,B[1][0].sup,B[1][0].sup]
            ax.fill(x_1,x_2,alpha=0.5,color="darkorange")

        ax.set_xlim((X[0][0].inf -0.5,X[0][0].sup +0.5))
        ax.set_ylim((X[1][0].inf -0.5,X[1][0].sup +0.5))
        ax.grid(color="lightgray")
        ax.plot((X[0][0].inf -0.5,X[0][0].sup +0.5),[0,0],color="black")
        ax.plot([0,0],(X[1][0].inf -0.5,X[1][0].sup +0.5),color="black")
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

def iterations_in_objective_space_plot(func: Callable[[obvec],float],grad: Callable[[obvec],obvec],cons: Union[Callable[[obvec],obvec],None],X: intvec,data: dict,iterations: list,columns: int=3,
                                       title: str='Iterations in objective space',dspace: bool=True,fname: Union[str,None]=None,**args) -> None:
    """Generates a tabular representation in which the objective space is shown for given iterations, 
    including the surface of the objective function, the associated optimal value approximation progress and optionally the decision space.
    The arguments have to be a real objective function 'func' with associated gradient 'grad', a vector-valued constraint 'cons' (or Python-value: None), 
    a box 'X' surrounding the feasible set, a dictionary containing the corresponding box progress for a given iteration, 
    a list of iterations to display, an optional number of columns, an optional title of the plot, an optional flag for plotting the decision space 'dspace', 
    an optional file name to save the plot and additional optional parameters to 'plt.figure'."""
    
    rows = -(-len(iterations)//columns)
    lb_f = centerd_forms(func,grad,X,direction="lower")[0]
    ub_f = centerd_forms(func,grad,X,direction="upper")[0]
    
    fig = plt.figure(**args,layout="constrained")
    fig.suptitle(title,fontsize="x-large")
    fig.supxlabel(" ")

    for u,k in enumerate(iterations):

        ax = fig.add_subplot(rows,columns,u+1,projection='3d')

        X_1, X_2 = np.meshgrid(np.linspace(X[0][0].inf -0.5,X[0][0].sup+0.5,round(X[0][0].sup - X[0][0].inf +1)*100),np.linspace(X[1][0].inf -0.5,X[1][0].sup +0.5,round(X[1][0].sup - X[1][0].inf +1)*100))
        
        if dspace:
            Box_X = mpatch.Rectangle((X[0][0].inf,X[1][0].inf), X[0].width[0][0] ,X[1].width[0][0],alpha=0.5)
            ax.add_patch(Box_X)
            art3d.pathpatch_2d_to_3d(Box_X, z=0, zdir="z")

            if callable(cons):
                for Z in cons((X_1,X_2)):
                    ax.contour(X_1,X_2,Z,levels=[0],colors="purple")

            for B in data[k]:
                Box_B = mpatch.Rectangle((B[0][0].inf,B[1][0].inf), B[0].width[0][0] ,B[1].width[0][0],alpha=0.5,color="royalblue")
                ax.add_patch(Box_B)
                art3d.pathpatch_2d_to_3d(Box_B, z=0, zdir="z")
        
        ax.plot_surface(X_1, X_2, func((X_1,X_2)), cmap="Reds",alpha=0.5)

        for B in data[k]:
            Box_B_h = mpatch.Rectangle((B[0][0].inf,B[1][0].inf), B[0].width[0][0] ,B[1].width[0][0],alpha=0.8,color="darkorange")
            ax.add_patch(Box_B_h)
            art3d.pathpatch_2d_to_3d(Box_B_h, z=func(B.midpoint()), zdir="z")

        ax.set_xlim((X[0][0].inf -0.5,X[0][0].sup +0.5))
        ax.set_ylim((X[1][0].inf -0.5,X[1][0].sup +0.5))
        ax.set_zlim((lb_f,ub_f))
        ax.grid(color="lightgray")
        ax.plot((X[0][0].inf -0.5,X[0][0].sup +0.5),[0,0],[0,0],color="black")
        ax.plot([0,0],(X[1][0].inf -0.5,X[1][0].sup +0.5),[0,0],color="black")
        ax.plot([0,0],[0,0],(lb_f,ub_f),color="black")
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