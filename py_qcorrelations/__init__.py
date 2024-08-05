from sympy.physics.quantum import TensorProduct
from sympy import Matrix, S, cos, eye, sqrt, exp, sin, I, symbols
from collections import defaultdict

theta, phi, psi, chi = symbols("theta phi psi chi", real=True)

# General Vector Tools
def gen_unit_vector(i, dim):
    zeros = Matrix.zeros(dim, 1)
    zeros[i] = 1
    
    return zeros


def gen_basis(dim):
    return tuple(gen_unit_vector(i, dim) for i in range(dim)) 


def conmutador(A, B):
    return A*B - B*A


def anticonmutador(A, B):
    return A*B + B*A


def is_diagonal(a):
    x, y = a.shape
    
    if x != y:
        return False
    
    for i in range(x):
        for j in range(x):
            if i == j: continue

            if a[i, j] != 0:
                return False
            
    return True


tensor = lambda: defaultdict(tensor)


def generar_latex_tensores(matrices, simbolo):
    from sympy import latex

    lineas = []
    for indice, matriz in enumerate(matrices):
        lineas.append('\[')
        lineas.append(f'{simbolo}_{{{indice}ij}} = {latex(matriz)}')
        lineas.append('\]')

    return '\n'.join(lineas)


# General Quantum States
from .su2 import base_su2
from .su3 import base_su3
base_qubit = gen_basis(2)
base_qutrit = gen_basis(3)

def density(v):
    return v*v.adjoint()

bb = lambda x, y: TensorProduct(base_qubit[x], base_qubit[y])
bt = lambda x, y: TensorProduct(base_qubit[x], base_qutrit[y])
tt = lambda x, y: TensorProduct(base_qutrit[x], base_qutrit[y])

werner_23 = lambda p: p*density(1/sqrt(2)*(bt(0, 0) + bt(1, 2))) + (1-p)/S(6)*eye(6)
werner_33 = lambda p: p*density(1/sqrt(3)*(tt(0, 0) + tt(1, 1) + tt(2, 2))) + (1-p)/S(9)*eye(9)
mixto_base_22 = lambda p: p*density(1/sqrt(2)*(bb(0, 0) + bb(1, 1))) + (1 - p)*density(bb(0, 0))
mixto_base_23 = lambda p: p*density(1/sqrt(2)*(bt(0, 0) + bt(1, 2))) + (1 - p)*density(bt(0, 0))
mixto_base_33 = lambda p: p*density(1/sqrt(3)*(tt(0, 0) + tt(1, 1) + tt(2, 2))) + (1-p)*density(tt(0, 0))

def bloch_params(m):
    x, _ = m.shape
    a_basis, b_basis = (base_su2, base_su3) if x == 6 else (base_su3, base_su3)

    return Matrix(
        [[(m*TensorProduct(la, lb)).trace() for lb in b_basis] for la in a_basis]
    )


def partial_trace(rho, n, dimensions):
    tensorite = [eye(i) for i in dimensions]
    result = Matrix.zeros(int(rho.shape[0]/dimensions[n]))
    
    for b in gen_basis(dimensions[n]):
        tensorite[n] = b
        u = TensorProduct(*tensorite)
        result += u.adjoint()*rho*u
        
    return result


measures_2 = [density(qb) for qb in base_qubit]
measures_3 = [density(qt) for qt in base_qutrit]

param_pj_2_1 = lambda theta, phi: density(cos(theta/2)*base_qubit[0] + exp(I*phi)*sin(theta/2)*base_qubit[1])
param_pj_2_2 = lambda theta, phi: density(sin(theta/2)*base_qubit[0] - exp(I*phi)*cos(theta/2)*base_qubit[1])
qubit_measures = [param_pj_2_1(theta, phi), param_pj_2_2(theta, phi)]

a = lambda theta, phi: S(1)/2 * sin(2*theta) * sin(phi)**2
b = lambda theta, phi: S(1)/2 * cos(theta) * sin(2*phi)
c = lambda theta, phi: S(1)/2 * sin(theta) * sin(2*phi)
d = lambda theta, phi: S(1)/2 * sin(2*theta) * cos(phi)**2

param_pj_3_1 = lambda theta, phi, psi, chi: Matrix([
    [cos(theta)**2 * sin(phi)**2, exp(-I*(psi - chi)) * a(theta, phi), exp(I*chi)*b(theta, phi)],
    [exp(I*(psi-chi)) * a(theta, phi), sin(theta)**2 * sin(phi)**2, exp(I*psi)*c(theta, phi)],
    [exp(-I*chi) * b(theta, phi), exp(-I*psi)*c(theta, phi), cos(phi)**2]
])

param_pj_3_2 = lambda theta, phi, psi, chi: Matrix([
    [cos(theta)**2 * cos(phi)**2, exp(-I*(psi - chi)) * d(theta, phi), -exp(I*chi)*b(theta, phi)],
    [exp(I*(psi-chi)) * d(theta, phi), sin(theta)**2 * cos(phi)**2, -exp(I*psi)*c(theta, phi)],
    [-exp(-I*chi) * b(theta, phi), -exp(-I*psi)*c(theta, phi), sin(phi)**2]
])

param_pj_3_3 = lambda theta, phi, psi, chi: Matrix([
    [sin(theta)**2, -1/2*exp(-I*(psi - chi)) * sin(2*theta), 0],
    [-1/2*exp(I*(psi - chi)) * sin(2*theta), cos(theta)**2, 0],
    [0, 0, 0]
])

qutrit_measures = [param_pj_3_1(theta, phi, psi, chi), param_pj_3_2(theta, phi, psi, chi), param_pj_3_3(theta, phi, psi, chi)]
