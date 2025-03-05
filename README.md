# pyimpBB - A branch-and-bound method using the improvement function in Python

This package provides the implementation of a novel branch-and-bound algorithm for the outer approximation 
of all global minimal points of a nonlinear constrained optimization problem using the improvement function, 
internally referred to as 'improved_BandB', to the corresponding publication 'The improvement function in 
branch-and-bound methods' by P. Kirst, S. Schwarze and O. Stein. 

## Installation:
The easiest and yet preferred way to install this package is to use <kbd>pip</kbd>. 
This allows you to install the package either directly from PyPi:
	
    python3 -m pip install --upgrade pip
	python3 -m pip install pyimpBB
    
or from a download of the source on PyPi:

    python3 -m pip install --upgrade pip
    python3 -m pip download pyimpBB
    python3 -m pip install *.whl

or github:

    python3 -m pip install --upgrade pip setuptools wheel
    git clone https://github.com/uisab/Python_Code.git
    python3 -m pip setup.py sdist bdist_wheel
    python3 -m pip install dist/*.whl

In case of problems with the required packages pyinterval (or crlibm) during installation, try <kbd>python3 -m pip install 'setuptools<=74.1.3'</kbd> 
or look out [here](https://github.com/taschini/pyinterval/issues). 

## Structure/ API:
The package consists of the four modules <kbd>pyimpBB.helper</kbd>, <kbd>pyimpBB.bounding</kbd>, 
<kbd>pyimpBB.solver</kbd> and <kbd>pyimpBB.analyzing</kbd>, each of which represents a core task 
of the implementation and provides Python classes and functions corresponding to their name. 
The following table gives an overview of these modules and their relevant classes and functions 
with a short description of each. For more detailed descriptions of the functionality, input and output, 
use <kbd>help(<module/class/function name>)</kbd>.
<table>
    <thead>
        <tr>
            <th>Modules</th>
            <th>Classes/ functions</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4 align="left" valign="top" >helper</td>
            <td valign="top">class obvec(tuple)</td>
            <td valign="top">A vector consisting of arbitrary objects with vector-valued support for all 
                elementary operations (+,-,*,/,&,|,...) as well as the scalar product (@).</td>
        </tr>
        <tr>
            <td valign="top">class obmat(tuple)</td>
            <td valign="top">A matrix consisting of arbitrary objects with matrix-valued support for all 
                elementary operations (+,-,*,/,&,|,...) as well as the matrix product (@).</td>
        </tr>
        <tr>
            <td valign="top">class intvec(obvec)</td>
            <td valign="top">A vector consisting of intervals from the pyinterval package, which supports 
                all elementary operations (+,-,*,/,&,|,...) interval-valued.</td>
        </tr>
        <tr>
            <td valign="top">exp(x), log(x), sin(x), cos(x), tan(x), sqrt(x)</td>
            <td valign="top">Refers to the corresponding mathematical function matching the input type.</td>
        </tr>
        <tr>
            <td rowspan=4 align="left" valign="top">bounding</td>
            <td valign="top">direct_intervalarithmetic(func, grad, hess, X, direction)</td>
            <td valign="top">Uses pur interval arithmetic to return an upper or lower bound of the real 
                function 'func' on the interval-vector 'X' in the form of an object-vector.</td>
        </tr>
         <tr>
            <td valign="top">centered_forms(func, grad, hess, X, direction)</td>
            <td valign="top">Uses centered forms to return an upper or lower bound of the real potentially 
                vector-valued function 'func' on the interval-vector 'X' in the form of an object-vector.</td>
        </tr>
         <tr>
            <td valign="top">optimal_centerd_forms(func, grad, hess, X, direction)</td>
            <td valign="top">Uses optimal centered forms to return an upper or lower bound of the real 
                function 'func' on the interval-vector 'X' in the form of an object-vector.</td>
        </tr>
         <tr>
            <td valign="top">aBB_relaxation(func, grad, hess, X, direction)</td>
            <td valign="top">Uses konvex relaxation via aBB method to return an upper or lower bound of the real 
                function 'func' on the interval-vector 'X' in the form of an object-vector.</td>
        </tr>
        <tr>
            <td rowspan=2 align="left" valign="top">solver</td>
            <td valign="top">improved_BandB(func, cons, X, bounding_procedure, grad=None, hess=None, cons_grad=[], 
                cons_hess=[], epsilon=0, epsilon_max=0.5, k_max=2500)</td>
            <td valign="top">Uses the improvement function in the course of a branch-and-bound approach to 
                provide an enclosure of the solution set of a nonlinear constrained optimization problem 
                with a given accuracy.</td>
        </tr>
        <tr>
            <td valign="top">improved_boxres_BandB(func, X, bounding_procedure, grad=None, hess=None, epsilon=0, 
                epsilon_max=0.5, k_max=2500)</td>
            <td valign="top">Uses the improvement function in the course of a branch-and-bound approach to 
                provide an enclosure of the solution set of a nonlinear box-constrained optimization problem 
                with a given accuracy.</td>
        </tr>
        <tr>
            <td rowspan=2 align="left" valign="top">analyzing</td>
            <td valign="top">iterations_in_decision_space_plot(func, X, data, iterations, cons=None, 
                title="Iterations in decision space", fname=None, columns=3, levels=None, mgres=100, 
                xylim=None, **args)</td>
            <td valign="top">Generates a tabular representation in which the decision space is shown for 
                given iterations, including level lines of the objective function, zero level lines of 
                the constraints, enclosing box X and associated approximation or decomposition progress.</td>
        </tr>
        <tr>
            <td valign="top">iterations_in_objective_space_plot(func, X, data, iterations, grad=None, cons=None,
                title='Iterations in objective space',fname=None, columns=3, dspace=True, mgres=100, 
                xyzlim=None, **args)</td>
            <td valign="top">Generates a tabular representation in which the objective space is shown for 
                given iterations, including the surface of the objective function, the associated optimal value 
                approximation progress and optionally the decision space.</td>
        </tr>
    </tbody>
</table>

## Application:
The use of this package will be shown and explained using a simple example. To do this, consider the optimization 
problem of the form

$$\min_x f(x) \quad s.t. \quad \omega(x) = \max\lbrace \omega_1(x), \ldots, \omega_4(x) \rbrace \leq 0,\quad\negthickspace x \in X$$

with nonempty box $X := ([0,3],[0,3])^\intercal \subseteq \mathbb{I}\negthinspace\mathbb{R}^2$ and continuously 
differentiable functions $f,\omega_i: \mathbb{R}^2 \rightarrow \mathbb{R}, i \in \lbrace 1, \ldots, 4\rbrace$ 
defined as

$$f(x) := x_1 + x_2,$$ 
$$\omega_1(x) := -(x_1^2 + x_2^2) +4, \quad \omega_3(x) := x_1 -x_2 -2,$$
$$\omega_2(x) := -x_1 +x_2 -2, \qquad\negthickspace \omega_4(x) := x_1^2 +x_2^2 -9.$$

This example problem should now be solved using the algorithm provided by this package and then analyzed using 
the representation functions also provided. To do this, it can first be modeled as follows using the class 
<kbd>intvec</kbd> from the module <kbd>pyimpBB.helper</kbd>.

    from pyimpBB.helper import intvec

    X = intvec([[0,3],[0,3]])

    def func(x):
        return x[0] + x[1]

    def omega_1(x):
        return -(x[0]**2 +x[1]**2) +4

    def omega_2(x):
        return -x[0] +x[1] -2

    def omega_3(x):
        return x[0] -x[1] -2

    def omega_4(x):
        return x[0]**2 +x[1]**2 -9

Here, <kbd>intvec</kbd> not only provides a simple way of instantiating a vector of intervals, but also some 
required properties, such as their width <kbd>width()</kbd> and splitting <kbd>split()</kbd>. As can be seen, 
it is instantiated using an $n$-element list of lists with exactly two entries, which correspond to the upper 
and lower bounds of the intervals. The interval objects used in this context come from the pyinterval package 
and have an interval-valued implementation of all elementary operations (+,-,*,/,&,...). 
The two classes <kbd>obvec</kbd> and <kbd>obmat</kbd> from the <kbd>pyimpBB.helper</kbd> module are used as follows 
to model the first and second derivatives that are required later depending on the selected bounding procedure.

    from pyimpBB.helper import obvec, obmat

    def grad(x):
        return obvec([1,1])
    def hess(x):
        return obmat([[0,0],[0,0]])

    def omega_1_grad(x):
        return obvec([-2*x[0],-2*x[1]])
    def omega_1_hess(x):
        return obmat([[-2,0],[0,-2]])

    def omega_2_grad(x):
        return obvec([-1,1])
    def omega_2_hess(x):
        return obmat([[0,0],[0,0]])

    def omega_3_grad(x):
        return obvec([1,-1])
    def omega_3_hess(x):
        return obmat([[0,0],[0,0]])

    def omega_4_grad(x):
        return obvec([2*x[0],2*x[1]])
    def omega_4_hess(x):
        return obmat([[2,0],[0,2]])

They represent a vector or a matrix of arbitrary Python objects and have a corresponding vector or matrix-valued 
implementation of all elementary operations (+,-,*,/,&,|,...) as well as the scalar or matrix product (@). In order to 
ensure the functionality of the required vector-valued interval arithmetic, its use should not be abandoned in favor 
of better known alternatives, such as <kbd>numpy.ndarray</kbd>. However, both support conversion to and from 
<kbd>numpy.ndarray</kbd>, which means that all functions of the numpy package can be used. It should be noted that 
the instantiation of <kbd>obmat</kbd> is done column-wise compared to <kbd>numpy.ndarray</kbd>. 
The following table shows the bounding procedures available in the <kbd>pyimpBB.bounding</kbd> module, along with 
their differentiability requirements.

<table>
    <thead>
        <tr>
            <th>Bounding procedures</th>
            <th>First derivative/ gradient</th>
            <th>Second derivative/ hessian</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td valign="top">direct_intervalarithmetic(func, grad, hess, X, direction)</td>
            <td valign="top">False</td>
            <td valign="top">False</td>
        </tr>
         <tr>
            <td valign="top">centered_forms(func, grad, hess, X, direction)</td>
            <td valign="top">True</td>
            <td valign="top">False</td>
        </tr>
         <tr>
            <td valign="top">optimal_centerd_forms(func, grad, hess, X, direction)</td>
            <td valign="top">True</td>
            <td valign="top">False</td>
        </tr>
         <tr>
            <td valign="top">aBB_relaxation(func, grad, hess, X, direction)</td>
            <td valign="top">True</td>
            <td valign="top">True</td>
        </tr>
    </tbody>
</table>

In order to be able to apply the algorithm designed for such problems in the form of the function 
<kbd>improved_BandB()</kbd> from the module <kbd>pyimpBB.solver</kbd> with a selected bound operation, such as 
<kbd>aBB_relaxation()</kbd>, it is still necessary to specify certain accuracies with regard to the feasibility 
(delta, delta_max) and optimality (epsilon, epsilon_max) of the solution as well as to define three auxiliary 
variables in the form of lists for clear transfer.

    from pyimpBB.bounding import aBB_relaxation
    from pyimpBB.solver import improved_BandB

    cons = [omega_1,omega_2,omega_3,omega_4]
    cons_grad = [omega_1_grad,omega_2_grad,omega_3_grad,omega_4_grad]
    cons_hess = [omega_1_hess,omega_2_hess,omega_3_hess,omega_4_hess]

    solution, k, save = improved_BandB(func, cons, X, bounding_procedure=aBB_relaxation, grad=grad, hess=hess, cons_grad=cons_grad, cons_hess=cons_hess, epsilon=0, delta=0, epsilon_max=0.5, delta_max=0.5, k_max=2500)

The return of this function consists of the actual solution of the algorithm as a list of <kbd>intvec</kbd>, 
the number of iterations required as an integer and a dictionary provided for analysis purposes, which documents 
the approximation progress of the algorithm for each iteration. With the help of the two functions 
<kbd>iterations_in_decision_space_plot</kbd> and <kbd>iterations_in_objective_space_plot</kbd> from the module 
<kbd>pyimpBB.analyzing</kbd> this progress can be displayed graphically, with both having an extensive range 
of options for influencing the resulting graphic. To use these functions, the data to be displayed must be extracted 
from the return of the function <kbd>improved_BandB()</kbd> as a dictionary with the iterations as keys and 
a selection of the iterations to be displayed as a list.

    from pyimpBB.analyzing import iterations_in_decision_space_plot

    data = dict(zip(save.keys(),[[Oi[0] for Oi in save[k][0]] for k in save]))
    iterations = list(data.keys())[::round(k/3)]
    
    iterations_in_decision_space_plot(func,X,data,iterations,cons=cons,columns=2,levels=[2,2.5],figsize=(8,6),facecolor="white")

![Representation of the approximation progress of the algorithm in the decision space of the test example for given iterations](https://github.com/uisab/pyimpBB/blob/master/doc_bsp_plot.png)

In this plot, the box $X$ (colored light blue), the feasible set $\Omega  := \lbrace x \in X \mid \omega(x) \leq 0 \rbrace$ 
(colored purple), the course of the objective function $f$ based on the level lines to the global minimum value $v$ 
and $v$ plus epsilon_max (red) and the iterative approximation progress of the algorithm (colored orange) 
are clearly visible. However, since such a graphical view of the results only proves to be useful for 
two-dimensional problems, the corresponding functions can only be applied to these. 
At the end of this example, for the theoretical background, reference is made to the already mentioned 
publication 'The improvement function in branch-and-bound methods' by P. Kirst, S. Schwarze and O. Stein.

## Miscellaneous
This package was created as part of a master thesis ('Analysis of a branch-and-bound method for nonlinear 
constrained optimization problems' by M. Rodestock) and we strive to provide a high-quality presentation to 
the best of our knowledge and belief. However, the author assumes no responsibility for the use of this package 
in any context. If you have suggestions for improvement or requests, the author asks for your understanding 
if he tends not to comply with them in the long term. Otherwise, enjoy this little package!

(written on 03.03.2025)