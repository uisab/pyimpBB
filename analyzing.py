'''This module contains analyzing and mapping functions for application to the solvers included in this package.'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

def iterations_in_decision_space_plot(func,cons,X,data,iterations,columns=3,title="Iterations in decision space",**args):
    """Generates a tabular representation in which the decision space is shown for given iterations, 
    including level lines of the objective function, zero level lines of the constraints, enclosing box X and associated approximation or decomposition progress.
    The arguments have to be a real objective function 'func', a vector-valued constraint 'cons', 
    a box 'X' surrounding the feasible set, a dictionary containing the corresponding box progress for a given iteration, 
    a list of iterations to display, the optional number of columns, the optional title of the plot and additional optional parameters to 'plt.figure'."""

    rows = -(-len(iterations)//columns)

    fig = plt.figure(**args,layout="constrained")
    fig.suptitle(title,fontsize="x-large",verticalalignment='center')
    fig.supxlabel(" ")

    for u,k in enumerate(iterations):
        ax = fig.add_subplot(rows,columns,u+1)

        ax.fill([X[0][0].inf,X[0][0].sup,X[0][0].sup,X[0][0].inf],[X[1][0].inf,X[1][0].inf,X[1][0].sup,X[1][0].sup],alpha=0.5, color="lightblue")

        X_1, X_2 = np.meshgrid(np.linspace(X[0][0].inf -0.5,X[0][0].sup+0.5,round(X[0][0].sup - X[0][0].inf +1)*100),np.linspace(X[1][0].inf -0.5,X[1][0].sup +0.5,round(X[1][0].sup - X[1][0].inf +1)*100))
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
    plt.show()