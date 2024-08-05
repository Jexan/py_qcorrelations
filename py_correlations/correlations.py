from collections import namedtuple
from sympy import symbols
from . import base_qubit, base_qutrit
from itertools import product
from sympy import Matrix, log, Abs
from sympy.physics.quantum import TensorProduct
import numpy as np
from scipy.optimize import minimize
from lib import *

theta, phi, psi, chi = symbols("theta phi psi chi", real=True)
var_spec = namedtuple("VarSpec", ["var", "initial_value", "lower_bound", "upper_bound"])
qutrit_specs = [var_spec(theta, 0, -np.pi, np.pi), var_spec(phi, 0, -np.pi, np.pi), var_spec(psi, 0, -np.pi/2, np.pi/2), var_spec(chi, 0, -np.pi, np.pi)]


# Negativity
def partial_transpose(m, dimension):
    x, y = m.shape

    if x == 4:
        basis_1, basis_2 = base_qubit, base_qubit
    elif x == 6:
        basis_1, basis_2 = base_qubit, base_qutrit
    elif x == 9:
        basis_1, basis_2 = base_qutrit, base_qutrit
    
    ordered_basis = tuple(TensorProduct(i, j) for i,j in product(basis_1, basis_2))
    component_map = tuple((i,j) for i in range(len(basis_1)) for j in range(len(basis_2)))
    index_to_component = {(i,j): TensorProduct(basis_1[i], basis_2[j]) for i in range(len(basis_1)) for j in range(len(basis_2))}
    new_m = Matrix.zeros(x, y)

    for i in range(x):
        for j in range(y):

            a_bits = list(component_map[i])
            b_bits = list(component_map[j])

            a_bits[dimension], b_bits[dimension] = b_bits[dimension], a_bits[dimension]

            new_a = index_to_component[tuple(a_bits)]
            new_b = index_to_component[tuple(b_bits)]

            new_m[ordered_basis.index(new_a), ordered_basis.index(new_b)] = m[i, j]

    return new_m


def negativity(m):
    eigen_vals = partial_transpose(m, 0).eigenvals()
    result = (sum(Abs(k)*v for k, v in eigen_vals.items()) - 1)/2

    return result

# MID
import numpy as np
from . import partial_trace, werner_33, werner_23, measures_2, measures_3, is_diagonal
import matplotlib.pyplot as plt

p, a = symbols("p a", real=True, positive=True)

def entropy(rho):
    try:
        ev = np.linalg.eigvals(np.array(rho).astype(np.cdouble))

        return -sum(i*np.log2(i) for i in ev if i)
    except Exception as e:
        print(rho.evalf().applyfunc(lambda x: round(x, 3)))
        raise e

def entropy_werner23(val):
    return -sum(i*np.log2(i) for i in werner_33(p).subs(p, val).eigenvals().keys() if i > 0.001).evalf()

def entropy_werner33(val):
    return -sum(i*np.log2(i) for i in werner_33(p).subs(p, val).eigenvals().keys() if i > 0.001).evalf()

def mutual_info(state, dims):
    state_a = partial_trace(state, 0, dims)
    state_b = partial_trace(state, 1, dims)

    return entropy(state_a) + entropy(state_b) - entropy(state)

def project_cannonical(rho):
    projectors_iter = product(measures_2, measures_3) if rho.shape[0] == 6 else product(measures_3, measures_3)
    projectors_iter = map(lambda x: TensorProduct(*x), projectors_iter)

    return sum((p*rho*p for p in projectors_iter), Matrix.zeros(rho.shape[0]))

def mid_cannonical(rho):
    dims = (2, 3) if rho.shape[0] == 6 else (3, 3)

    assert is_diagonal(partial_trace(rho, 1, dims))
    assert is_diagonal(partial_trace(rho, 0, dims))
    
    return mutual_info(rho, dims) - mutual_info(project_cannonical(rho), dims)


def plot_mid_2d(rho, symbol=p, n_params=100, model_on=False):
    p_vs = np.linspace(0, 1, n_params)

    values = []
    for p_v in p_vs:
        values.append(mid_cannonical(rho.subs(symbol, p_v)).real)

    # Plot the surface.
    
    plot_ax = plot_discord_2d(rho, qutrit_specs, symbol, n_params, 5, file_path=None, show_neg=True, model_on=True)
    plot_ax.plot(p_vs, values, label="MID")
    if model_on:
        return plot_ax
    else:
        plot_ax.legend()
        plt.show()
        


def plot_mid(rho, vars=(a, p), n=10, file_path=None):
    values = []
    for a_v in np.linspace(0, 1, n):
        for p_v in np.linspace(0, 1, n):
            values.append(mid_cannonical(rho.subs([(vars[0], a_v), (vars[1], p_v)])).real)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)

    # Plot the surface.
    ax.plot_surface(X, Y, np.array(values).reshape(n, n), cmap="viridis", cstride=1, rstride=1)
    plt.rcParams['figure.figsize'] = 12, 12
    plt.rc('axes', labelsize=20)
    ax.set_xlabel(f'${vars[1]}$')
    ax.set_ylabel(f'${vars[0]}$')
    ax.set_zlabel(f'$f({vars[1]}, {vars[0]})$')
    
    plt.autoscale()
    fig.savefig(file_path, format="svg", bbox_inches = 'tight', dpi=200)

# AMID

from itertools import product
k, l, xi, varepsilon = symbols("k l, xi, varepsilon")
def plot_amid2d(rho, var, parameters=100):
    rho_i_APB = (
        TensorProduct(Pj, Pk)*rho*TensorProduct(Pj, Pk) for Pj, Pk in product(qutrit_measures, (pj.subs([(theta, k), (phi, l)] for pj in qubit_measures)) )
    )

    total_op = sum(rho_i_APB, Matrix.zeros(6))

    p_vs = np.linspace(0, 1, parameters)
    min_discords = []

    for p_v in p_vs:
        def entropy_postmeasure(x_v):
            th, ph, ps, ch, x_s, y_s = x_v

            proyector1_sub = total_op.subs([
                (var, p_v), (theta, th), (phi, ph), (psi, ps), (chi, ch),
                (k, x_s), (l, y_s)
            ]).applyfunc(lambda x: x.evalf())
            
            return entropy(proyector1_sub)

        bnds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
        min_entropy = minimize(entropy_postmeasure, (0, 0, 0, 0, 0, 0), bounds=bnds, method='TNC')

        min_discords.append(min_entropy.fun.real - entropy(rho.subs(var, p_v)))

        print(p_v)
        
    ax = plot_mid_2d(rho, var, parameters, True)

    ax.plot(np.linspace(0, 1, parameters),[x.real for x in min_discords], label="AMID")
    ax.legend()
    plt.show()


def plot_amid2d_2qt(rho, var, parameters=100):
    rho_i_APB = (
        TensorProduct(Pj, Pk)*rho*TensorProduct(Pj, Pk) for Pj, Pk in product(
            qutrit_measures, 
            (pj.subs([(theta, k), (phi, l), (chi, xi), (psi, varepsilon)]) for pj in qutrit_measures) 
        )
    )

    total_op = sum(rho_i_APB, Matrix.zeros(9))

    p_vs = np.linspace(0, 1, parameters)
    min_discords = []

    for p_v in p_vs:
        def entropy_postmeasure(x_v):
            th, ph, ps, ch, th_2, phi_2, ps_2, ch_2 = x_v

            proyector1_sub = total_op.subs([
                (var, p_v), (theta, th), (phi, ph), (psi, ps), (chi, ch),
                (k, th_2), (l, phi_2), (varepsilon, ps_2), (xi, ch_2)
            ]).applyfunc(lambda x: x.evalf())
            
            return entropy(proyector1_sub)

        bnds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi))
        min_entropy = minimize(entropy_postmeasure, (0, 0, 0, 0, 0, 0, 0, 0), bounds=bnds, method='TNC')

        min_discords.append(min_entropy.fun.real - entropy(rho.subs(var, p_v)))

        print(p_v)
        
    ax = plot_mid_2d(rho, var, parameters, True)

    ax.plot(np.linspace(0, 1, parameters),[x.real for x in min_discords], label="AMID")
    ax.legend()
    plt.show()

# QD

def get_discord(rho, n_params, symbolic_vars, parameter_var):
    is_qutrit = rho.shape[0] == 9
    projectors = qutrit_measures
    projector_dual_op = [TensorProduct(eye(3 if is_qutrit else 2), i) for i in projectors]
    
    rho_i_APB_nonnormalized = tuple(
        (pj*rho*pj) for pj in projector_dual_op
    )

    p_i = tuple(
        i.trace() for i in rho_i_APB_nonnormalized
    )

    rho_i_APB = tuple(
        1/p_j * rho_j for p_j, rho_j in zip(p_i, rho_i_APB_nonnormalized)
    )

    p_vs = np.linspace(0, 1, n_params)
    min_discords = []
    dims = (3, 3) if is_qutrit else (2, 3)

    symbol_vars = tuple(i.var for i in symbolic_vars)
    bounds = tuple((i.lower_bound, i.upper_bound) for i in symbolic_vars)
    initial_values = tuple(i.initial_value for i in symbolic_vars)

    for p_v in p_vs:
        min_entropy = float("inf")
        
        def entropy_postmeasure(x):
            current_entropy = 0

            for j, proyector in enumerate(rho_i_APB):
                proyector_subs = proyector.subs(parameter_var, p_v).subs(list(zip(symbol_vars, x))).subs(np.nan, 0)
                evs = proyector_subs.eigenvals()
                p_j = p_i[j].subs(parameter_var, p_v).subs(list(zip(symbol_vars, x)))

                for ev, mult in evs.items():
                    if complex(ev).imag:
                        ev = abs(ev)
                    if ev < 0: 
                        ev = 0
                    if not ev:
                        continue
                    
                    if abs(ev) > 0.00001:
                        term = mult*ev*p_j*log(ev, 2)
                        if term > 0:
                            continue
                        current_entropy += (-term)


            return_value = complex(S(current_entropy).evalf())
            return return_value.real
                    
        min_entropy = minimize(entropy_postmeasure, initial_values, bounds=bounds, method='TNC')

        min_discords.append(
            entropy(partial_trace(rho.subs(parameter_var, p_v), 0, dims)) - 
            entropy(rho.subs(parameter_var, p_v)) + 
            min_entropy.fun
        )

    return min_discords


def plot_discord_2d(rho, specs, param, n, width, file_path=None, show_neg=False, model_on=False):
    max_ent_discord = get_discord(rho, n, specs, param)

    fig, ax = plt.subplots(figsize=(1.5*width, width))
    ax.plot(np.linspace(0, 1, n), max_ent_discord, label='Discordia Cuántica')
    plt.rc('axes', labelsize=14)
    ax.set_ylabel(f'$f({param})$')
    ax.set_xlabel(f'${param}$')
    
    if show_neg:
        from sympy import plot
        a = plot(negativity(rho), (param, 0, 1), show=False)
        points = a[0].get_points() 
        ax.plot(points[0], points[1], label='Negatividad')
    
    if not model_on:
        ax.legend()
        plt.show()
    else: 
        return ax
    
    if file_path:
        fig.savefig(file_path, format="svg", bbox_inches = 'tight', dpi=200)


def plot_discord_3d(rho, specs, params, n, file_path):
    discords_per_a = []
    for a_v in np.linspace(0, 1, n):
        discords_per_a.append(get_discord(rho.subs(params[0], a_v), n, specs, params[1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    Z = np.array([i.real for j in discords_per_a for i in j]).reshape(n, n)
    
    ax.plot_surface(X, Y, Z, cmap="viridis", cstride=1, rstride=1)
    
    plt.rcParams['figure.figsize'] = 12, 12
    plt.rc('axes', labelsize=20)
    ax.set_xlabel(f'${params[1]}$')
    ax.set_ylabel(f'${params[0]}$')
    ax.set_zlabel(r'$f(a, p)$')
    fig.savefig(file_path, format="svg", bbox_inches = 'tight', dpi=200)
    
    plt.show()

    

# Relative Entropy of Entanglement
# separable_23 = TensorProduct(cos(theta/2)*base_qubit[0] + exp(I*phi)*sin(theta/2)*base_qubit[1], exp(I*c)*sin(a)*cos(b)*base_qutrit[0] + exp(I*d)*sin(a)*sin(b)*base_qutrit[1] + cos(a)*base_qutrit[2])
# separable_density_23 = density(separable_23)


# from lib.correlations import var_spec, rel_entropy
# from scipy.optimize import minimize


def rel_entropy(rho, sigma):
    return (rho*(m_log(rho) - m_log(sigma))).trace()

def m_log(rho):
    V, D = rho.diagonalize()
    if V.det() == 0:
        print(V, D)
        raise Exception
    
    return V*D.applyfunc(lambda x: log(x, 2) if x else 0)*V.inv()

# def optimize_rel_entr_23(rho, n_params, symbolic_vars, parameter_var):
#     is_qutrit = rho.shape[0] == 9
#     general_separable = separable_density_23 if not is_qutrit else None
    
#     p_vs = np.linspace(0, 1, n_params)
#     min_entropies = []

#     symbol_vars = tuple(i.var for i in symbolic_vars)
#     bounds = tuple((i.lower_bound, i.upper_bound) for i in symbolic_vars)
#     initial_values = tuple(i.initial_value for i in symbolic_vars)

#     for p_v in p_vs:
#         min_entropy = float("inf")
        
#         def rel_entropy_get(x):
#             general_subs = general_separable.subs(parameter_var, p_v).subs(list(zip(symbol_vars, x)))
#             rho_subs = rho.subs(parameter_var, p_v)
            
#             general_diag = general_subs.diagonal()
#             rho_diag = rho_subs.diagonal()
            
#             for i, j in zip(general_diag, rho_diag):
#                 if (not i) and j:
#                     return float("inf")
            
#             return round(complex(rel_entropy(rho_subs, general_subs).evalf()).real, 5)
                    
#         min_entropy = minimize(rel_entropy_get, initial_values, bounds=bounds, method='TNC')

#         min_entropies.append(min_entropy.fun)
        
#         print(p_v)

#     return min_entropies
    