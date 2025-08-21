from pyimpBB.helper import obmat, obvec, intvec, exp, log, sin, cos, tan, sqrt
from pyimpBB.bounding import optimal_centerd_forms, centerd_forms, aBB_relaxation, direct_intervalarithmetic
import cProfile
from pyimpBB.analyzing import iterations_in_decision_space_plot
from pyimpBB.solver import analysed_improved_BandB, improved_BandB, analysed_improved_BandB_3
from pyimpBB.solverv2 import ICGO
import cloudpickle

with open("Testbed/TP1.pkl", "rb") as f:
    TP = cloudpickle.load(f)
X = intvec([[0,4],[0,4]])

names = ['1','3','41','42','5','61','62']#2 fehlt
time_v1 =[]
time_v2 =[]
it_v1 = []
it_v2 = []
solution_v1 = []
solution_v2 = []
names_full = []

for name in names:
    print(name)
    path = "Testbed/TP"+name+".pkl"
    names_full.append("TP"+name)
    with open(path, "rb") as f:
        TP = cloudpickle.load(f)
    solution, y_best, k, t = improved_BandB(TP['func'], TP['cons'], X, bounding_procedure=optimal_centerd_forms,
                                                           grad=TP['grad'], hess=TP['hess'], cons_grad=TP['cons_grad'],
                                                           cons_hess=TP['cons_hess'], epsilon=1e-5, delta=0, epsilon_max=0.5,
                                                           delta_max=0.5)
    time_v1.append(t)
    it_v1.append(k)
    solution_v1.append(len(solution))

    solution, y_best, k, t = ICGO(TP['func'], TP['cons'], X, bounding_procedure=optimal_centerd_forms,
                                                           grad=TP['grad'], hess=TP['hess'], cons_grad=TP['cons_grad'],
                                                           cons_hess=TP['cons_hess'], epsilon=1e-5, delta=0, epsilon_max=0.5,
                                                           delta_max=0.5, max_time=180)

    time_v2.append(t)
    it_v2.append(k)
    solution_v2.append(len(solution))

print(time_v1)
print(time_v2)
print(it_v1)
print(it_v2)
print(solution_v1)
print(solution_v2)

change = []
for (i,item) in enumerate(time_v1):
    change.append(100*((time_v2[i]/time_v1[i])-1))
print(change)


