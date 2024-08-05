from sympy import Matrix, conjugate, symbols, sqrt, sin, cos, pi

rho_11, rho_22, rho_33, rho_44, rho_55, rho_66, rho_61, rho_16, rho_51, rho_15, rho_42, rho_24, rho_43, rho_34, rho_62, rho_26, rho_35, rho_53 = symbols(
    'rho11 rho22 rho33 rho44 rho55 rho66 rho61 rho16 rho51 rho15 rho42 rho24 rho_43 rho_34 rho_62 rho_26 rho_35 rho_53'
)

tgx_23_1 = Matrix([
    [rho_11, 0, 0, 0, rho_15, 0],
    [0, rho_22, 0, rho_24, 0, 0],
    [0, 0, rho_33, 0, 0, 0],
    [0, rho_42, 0, rho_44, 0, 0],
    [rho_51, 0, 0, 0, rho_55, 0],
    [0, 0, 0, 0, 0, rho_66]
])
tgx_23_2 = Matrix([
    [rho_11, 0, 0, 0, 0, rho_16],
    [0, rho_22, 0, 0, 0, 0],
    [0, 0, rho_33, rho_34, 0, 0],
    [0, 0, rho_43, rho_44, 0, 0],
    [0, 0, 0, 0, rho_55, 0],
    [rho_61, 0, 0, 0, 0, rho_66]  
])
tgx_23_3 = Matrix([
    [rho_11, 0, 0, 0, 0, 0],
    [0, rho_22, 0, 0, 0, rho_26],
    [0, 0, rho_33, 0, rho_35, 0],
    [0, 0, 0, rho_44, 0, 0],
    [0, 0, rho_53, 0, rho_55, 0],
    [0, rho_62, 0, 0, 0, rho_66]  
])

tgx = tgx_23_2.subs(rho_61, conjugate(rho_16)).subs(rho_43, conjugate(rho_34))
x, z = symbols('x z', real=True)

rho_22x = rho_55x = rho_11x = rho_66x = rho_33x = x/5
rho_44x = 1 - rho_11x - rho_33x - rho_55x - rho_66x - rho_22x
rho_16x = sqrt(rho_11x*rho_66x)
rho_34x = sqrt(rho_33x*rho_44x)
uniparametric_1 = tgx.subs(rho_33, rho_33x).subs(rho_44, rho_44x).subs(rho_16, rho_16x).subs(rho_34, rho_34x).subs(rho_11, rho_11x).subs(rho_22, rho_22x).subs(rho_55, rho_55x).subs(rho_66, rho_66x).subs(x, z)

rho_22x = rho_55x = rho_33x =  (1+sin(2*pi*x))/10
rho_66x = rho_11x = (1+cos(2*pi*x))/10
rho_44x = 1 - rho_11x - rho_33x - rho_55x - rho_66x - rho_22x

rho_16x = sqrt(rho_11x*rho_66x)
rho_34x = sqrt(rho_33x*rho_44x)

uniparametric_2 = tgx.subs(rho_33, rho_33x).subs(rho_44, rho_44x).subs(rho_16, rho_16x).subs(rho_34, rho_34x).subs(rho_11, rho_11x).subs(rho_22, rho_22x).subs(rho_55, rho_55x).subs(rho_66, rho_66x).subs(x, z)
uniparametric_2