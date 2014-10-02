from dolfin import *
from pylab import deg2rad,plot,show,linspace,ones,zeros,copy,array 

parameters['form_compiler']['quadrature_degree'] = 2
parameters['form_compiler']['precision'] = 30
#parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = 'quadrature'

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
# TIME
minute = 60.0
hour = 60*minute
day = 24*hour
year = 365*day

# CONSTANTS
rho = 910.
g = 9.81
n = 3.0
A = 1e-16
b = 1e-16**(-1./n)

k = 2.1*year
Cp = 2009.
kappa = k/(rho*Cp)

q_geo = 0.042*year

Bc = 3.61e-13*year
Bw = 1.73e3*year
Qc = 6e4
Qw = 13.9e4
Rc = 8.314

# TIME STEP AND REGULARIZATION
eps_reg = 1e-10
dt = 1.0
thklim = 1.0

# GEOMETRY AND INPUT DATA
class Beta2(Expression):
  def eval(self,values,x):
    values[0] = 1e10

# NUMBER OF DOFS PER VELOCITY COMPONENT
#polynomial_degree = 4
#polynomial_degree = 3
polynomial_degree = 2
N_T = 8
deltax = 1./(N_T-1.)
gamma = 8.7e-4
sigmas = linspace(0,1,N_T,endpoint=True)

#########################################################
#################       MESH        #####################
#########################################################
mesh = Mesh("isunn_mesh_2000m.xml")
boundaries = MeshFunction('size_t', mesh, 'isunn_boundary_indicators_2000m.xml')
normal = FacetNormal(mesh)
h = CellSize(mesh)
hmin = project(h,FunctionSpace(mesh,'CG',1)).vector().min()

# Interior numbering
GROUNDED = 0 
FLOATING = 1 
domains     = CellFunction ("size_t", mesh)
domains.set_all(GROUNDED)      # Default will be floating points.

# Boundary numbering
DIVIDE       = 0
MARGIN       = 1
FREESLIP     = 2
NOT_ASSIGNED = 3
SHELF        = 4

#dx = Measure("dx")[domains]
#ds = Measure("ds")[boundaries]


# FUNCTION SPACES
Q = FunctionSpace(mesh,"CG",1) # SCALAR
V = MixedFunctionSpace([Q]*2*polynomial_degree) # VELOCITY
Z = MixedFunctionSpace([Q]*N_T) # TEMPERATURE

#########################################################
#################       DATA        #####################
#########################################################
data_dir = './isunnguata_2000m/'
B = Function(Q,data_dir+'Bed.xml')
H = Function(Q,data_dir+'H.xml')
adot = Function(Q,data_dir+'adot.xml')
# Clean up a bad interpolation artifact on boundary
adot.vector()[adot.vector()>-.25] = -3.
T_s = Function(Q,data_dir+'Ts.xml')

beta2 = interpolate(Beta2(),Q)

#########################################################
#################    FUNCTIONS      #####################
#########################################################
U = Function(V)
Phi = TestFunction(V)
dU = TrialFunction(V)

dH = TrialFunction(Q)
xsi = TestFunction(Q)
H0 = Function(Q)
#H.vector()[:] = thklim
#H0.vector()[:] = thklim
H0.assign(H)
S = B+H

T_ = Function(Z)
Psi = TestFunction(Z)
dT = TrialFunction(Z)

T0_ = project(as_vector([T_s]*N_T),Z)

# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

# SIMILAR TO ABOVE, BUT FOR CALCULATION OF FINITE DIFFERENCE QUANTITIES.
class VerticalFDBasis(object):
    def __init__(self,u,deltax):
        self.u = u 
        self.deltax = deltax
        self.coef = coef

    def __call__(self,i):
        return self.u[i]

    def eval(self,s):
        fl = max(sum(s>sigmas)-1,0)
        dist = s - sigmas[fl]
        return self.u[fl]*(1 - dist/self.deltax) + self.u[fl+1]*dist/self.deltax

    def ds(self,i):
        return (self.u[i+1] - self.u[i-1])/(2*self.deltax)

    def d2s(self,i):
        return (self.u[i+1] - 2*self.u[i] + self.u[i-1])/(self.deltax**2)

    def dx(self,i,x):
        return self.u[i].dx(x)        

# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

# METRICS FOR COORDINATE TRANSFORM
def dsdx(s):
    return 1./H*(S.dx(0) - s*H.dx(0))

def dsdy(s):
    return 1./H*(S.dx(1) - s*H.dx(1))

def dsdz(s):
    return -1./H


# COEFFICIENTS AND LISTS NECESSARY FOR INSTANTIATION OF VERTICAL BASIS FUNCTIONS.  THESE 
# ARE THE SPECTRAL ELEMENT TEST FUNCTIONS!

# LEGENDRE-4
#coef = [lambda s:1.0, lambda s:0.5*(3*s**2-1.0), lambda s:0.125*(35*s**4-30*s**2+3), lambda s:1./16.*(231*s**6 - 315*s**4 + 105*s**2 - 5.)]
#dcoef = [lambda s:0.0, lambda s:0.5*(6*s), lambda s:0.125*(140*s**3-60*s), lambda s:1/16.*(1368*s**5 - 1260*s**3 + 210*s)]
#icoef = [lambda s:s, lambda s:0.5*(s**3 - s),lambda s:0.125*(7*s**5 - 10*s**3 + 3*s), lambda s:1./16.*(33*s**7 - 63*s**5 + 35*s**3 - 5*s)]

#u_ = [U[0],U[2],U[4],U[6]]
#v_ = [U[1],U[3],U[5],U[7]]
#phi_ = [Phi[0],Phi[2],Phi[4],Phi[6]]
#psi_ = [Phi[1],Phi[3],Phi[5],Phi[7]]

# LEGENDRE-3
#coef = [lambda s:1.0, lambda s:0.5*(3*s**2-1.0), lambda s:0.125*(35*s**4-30*s**2+3)]
#dcoef = [lambda s:0.0, lambda s:0.5*(6*s), lambda s:0.125*(140*s**3-60*s)]

#u_ = [U[0],U[2],U[4]]
#v_ = [U[1],U[3],U[5]]
#phi_ = [Phi[0],Phi[2],Phi[4]]
#psi_ = [Phi[1],Phi[3],Phi[5]]

# ANSATZ    
coef = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
dcoef = [lambda s:0.0, lambda s:5*s**3]

coef_t = [lambda s:1.0, lambda s:0.0]
dcoef_t = [lambda s:0.0, lambda s:0.0]

u_ = [U[0],U[2]]
v_ = [U[1],U[3]]
phi_ = [Phi[0],Phi[2]]
psi_ = [Phi[1],Phi[3]]

u = VerticalBasis(u_,coef,dcoef)
v = VerticalBasis(v_,coef,dcoef)

phi = VerticalBasis(phi_,coef_t,dcoef_t)
psi = VerticalBasis(psi_,coef_t,dcoef_t)

T = VerticalFDBasis(T_,deltax)
T0 = VerticalFDBasis(T0_,deltax)

def A_v(T):
    return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))

def epsilon_dot(s):
    return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                + eps_reg)

def eta_v(s):
    return A_v(T0.eval(s))**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))

def membrane_xx(s):
    return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*H*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

def membrane_xy(s):
    return (phi.dx(s,1) + phi.ds(s)*dsdy(s))*H*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yx(s):
    return (psi.dx(s,0) + psi.ds(s)*dsdx(s))*H*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yy(s):
    return (psi.dx(s,1) + psi.ds(s)*dsdy(s))*H*eta_v(s)*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

def shear_xz(s):
    return dsdz(s)**2*phi.ds(s)*H*eta_v(s)*u.ds(s)

def shear_yz(s):
    return dsdz(s)**2*psi.ds(s)*H*eta_v(s)*v.ds(s)

def tau_dx(s):
    return rho*g*H*S.dx(0)*phi(s)

def tau_dy(s):
    return rho*g*H*S.dx(1)*psi(s)

def w(s):
    w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1.)
    w_2 = (U[2].dx(0) + U[3].dx(1))*(s**(n+2) - s)/(n+1) + (n+2)/H*U[2]*(1./(n+1)*(s**(n+1) - 1.)*S.dx(0) - 1./(n+1)*(s**(n+2) - 1.)*H.dx(0)) + (n+2)/H*U[3]*(1./(n+1)*(s**(n+1) - 1.)*S.dx(1) - 1./(n+1)*(s**(n+2) - 1.)*H.dx(1))
    return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1./dsdz(s)*(w_0 + w_2) 

# SIA DIFFUSION COEFFICIENT INTEGRAL TERM.
def sia_int(s):
    return A_v(T.eval(s))*s**(n+1)

# O(4)
points = array([0.0,0.4688,0.8302,1.0])
weights = array([0.4876/2.,0.4317,0.2768,0.0476])
# O(6)
#points = array([1.0,0.89976,0.677186,0.36312,0.0])
#weights = array([0.02778,0.1654595,0.274539,0.3464285,0.371519/2.])
# O(8)
#points = array([1,0.934001,0.784483,0.565235,0.295758,0])
#weights = array([0.0181818,0.10961,0.18717,0.248048,0.28688,0.300218/2.])

vi = VerticalIntegrator(points,weights)

### MOMENTUM BALANCE ###

R_x = - vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) - phi(1)*beta2*u(1) - vi.intz(tau_dx)
R_y = - vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) - psi(1)*beta2*v(1) - vi.intz(tau_dy)

S_x = phi_[1]*(eta_v(1.0)*(4*u.dx(1.0,0) + 2*v.dx(1.0,1))*B.dx(0) + eta_v(1.0)*(u.dx(1.0,1) + v.dx(1.0,0))*B.dx(1) - eta_v(1.0)*dsdz(1.0)*u.ds(1.0) + beta2*u(1.0))
S_y = psi_[1]*(eta_v(1.0)*(u.dx(1.0,1) + v.dx(1.0,0))*B.dx(0) + eta_v(1.0)*(2*u.dx(1.0,0) + 4*v.dx(1.0,1))*B.dx(1) - eta_v(1.0)*dsdz(1.0)*v.ds(1.0) + beta2*v(1.0))

# SIA

R = (R_x + R_y)*dx + (S_x + S_y)*dx
#R = replace(R,{U:dU})
J = derivative(R,U,dU)

### MASS BALANCE ###
ubar_c = Function(Q)
vbar_c = Function(Q)

#D = 2.*(rho*g)**n*A/(n+2.)*H**(n+2)*dot(grad(S),grad(S))**((n-1.)/2.)
D = 2.*(rho*g)**n*H**(n+2)*dot(grad(S),grad(S))**((n-1.)/2.)*vi.intz(sia_int) + rho*g*H**2/beta2

h = CellSize(mesh)
ubar = U[0]
vbar = U[1]

ubar_si = -D/H*S.dx(0)
vbar_si = -D/H*S.dx(1)

M = dH*xsi*dx
ubar_proj = (ubar-ubar_si)*xsi*dx
vbar_proj = (vbar-vbar_si)*xsi*dx

R_thick = ((H-H0)/dt*xsi + D*dot(grad(S),grad(xsi)) + xsi*(Dx(ubar_c*H,0)+Dx(vbar_c*H,1)) - adot*xsi)*dx
J_thick = derivative(R_thick,H,dH)


### ENERGY BALANCE ### 
R_T = 0

Tm = as_vector([273.15 - gamma*sigma*H for sigma in sigmas])
Tmb = 273.15 - gamma*H

for i in range(N_T):
    # SIGMA COORDINATE
    s = i/(N_T-1.0)

    # EFFECTIVE VERTICAL VELOCITY
    w_eff = u(s)*dsdx(s) + v(s)*dsdy(s) + w(s)*dsdz(s) + 1./H*(1.-s)*((H-H0)/dt)
 
    # STRAIN HEAT
    Phi_strain = (2*n)/(n+1)*2*eta_v(s)*epsilon_dot(s)
 
    # STABILIZATION SCHEME
    Umag = sqrt(u(s)**2 + v(s)**2 + 1e-3)
    tau = h/(2*Umag)
    Psihat = Psi[i] + tau*(u(s)*Psi[i].dx(0) + v(s)*Psi[i].dx(1))

    # TIME DERIVATIVE
    dTdt = (T(i) - T0(i))/dt

    # SURFACE BOUNDARY
    if i==0:
        R_T += Psi[i]*(T(i) - T_s)*dx
    # BASAL BOUNDARY
    elif i==(N_T-1):
        R_T += dTdt*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
        R_T += -w_eff*q_geo/(rho*Cp*kappa*dsdz(s))*Psi[i]*dx
        f = (q_geo + 0.5*beta2*(u(s)**2 + v(s)**2))/(rho*Cp*kappa*dsdz(s))
        R_T += -2.*kappa*dsdz(s)**2*((T(N_T-2) - T(N_T-1))/(deltax**2) - f/deltax)*Psi[i]*dx
    # INTERIOR
    else:
        R_T += dTdt*Psi[i]*dx
        R_T += -kappa*dsdz(s)**2.*T.d2s(i)*Psi[i]*dx
        R_T += w_eff*T.ds(i)*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 

# PRETEND THIS IS LINEAR (A GOOD APPROXIMATION IN THE TRANSIENT CASE)
R_T = replace(R_T,{T_:dT})

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

#Define variational solver for the momentum problem
momentum_problem = NonlinearVariationalProblem(R,U,J=J,form_compiler_parameters=ffc_options)
momentum_solver = NonlinearVariationalSolver(momentum_problem)
momentum_solver.parameters['nonlinear_solver'] = 'newton'
momentum_solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
momentum_solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
momentum_solver.parameters['newton_solver']['absolute_tolerance'] = 1e7
momentum_solver.parameters['newton_solver']['maximum_iterations'] = 20
momentum_solver.parameters['newton_solver']['error_on_nonconvergence'] = False
momentum_solver.parameters['newton_solver']['linear_solver'] = 'mumps'

#Define variational solver for the mass problem
mass_problem = NonlinearVariationalProblem(R_thick,H,J=J_thick,form_compiler_parameters=ffc_options)
mass_solver = NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'

mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['maximum_iterations'] = 20
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'

l_thick = project(Constant(thklim),Q)
u_thick = project(Constant(1e4),Q)

t = 0.0
t_end = 500.0

Us = project(as_vector([u(0.),v(0.0)]))
ww = project(w(0),Q)

results_dir = '/home/jessej/model_output/isunnguata/'

Ufile = File(results_dir + 'U.pvd')
wfile = File(results_dir + 'w.pvd')
Hfile = File(results_dir + 'H.pvd')
Sfile = File(results_dir + 'S.pvd')
Bfile = File(results_dir + 'B.pvd')
Hxfile = File(results_dir + 'H.xml')
Tbfile = File(results_dir + 'Tb.pvd')
Tsfile = File(results_dir + 'Ts.pvd')
Txfile = File(results_dir + 'T.xml')

Bfile << B

Ts_ = Function(Q)
Tb_ = Function(Q)
S_  = Function(Q)

#UNCOMMENT THESE IF YOU WANT TO START FROM WHERE YOU LEFT OFF.
#Hxfile >> H
#Hxfile >> H0

#Txfile >> T_
#Txfile >> T0_
# THOUGH YOU HAVE TO FILL THIS ONE IN YOURSELF.
#t = 20400

T_tol = 10.0

# Temp spinup loop:
for t_f in range(200):
    # STARTING THE VELOCITY SOLVE FROM A NON-CONVERGED INITIAL GUESS, THOUGH SLOWER, IS MUCH MORE STABLE
    U.vector()[:] = 0.0
    momentum_solver.solve()

    # Find corrective velocities
    solve(M==ubar_proj,ubar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)
    solve(M==vbar_proj,vbar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)

    # SOLVE TEMPERATURE
    solve(lhs(R_T)==rhs(R_T),T_,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)    
    # Update temperature field
    T_melt = project(Tm)
    Tb_m = project(Tmb)
    Tb_temp = project(T_[N_T-1],Q)

    beta2.vector()[Tb_temp.vector()>(Tb_m.vector() - T_tol)] = 7e2 # EISMINT II 1e3 
    beta2.vector()[Tb_temp.vector()<=(Tb_m.vector() - T_tol)] = 1e9

    T_.vector()[T_.vector()>T_melt.vector()] = T_melt.vector()[T_.vector()>T_melt.vector()]

    # UPDATE PREVIOUS TIME STEP
    T0_.interpolate(T_)

while t<t_end:
    # STARTING THE VELOCITY SOLVE FROM A NON-CONVERGED INITIAL GUESS, THOUGH SLOWER, IS MUCH MORE STABLE
    U.vector()[:] = 0.0
    momentum_solver.solve()

    # Find corrective velocities
    solve(M==ubar_proj,ubar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)
    solve(M==vbar_proj,vbar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)

    # SOLVE MASS CONS.
    mass_solver.solve(l_thick,u_thick)

    # SOLVE TEMPERATURE
    solve(lhs(R_T)==rhs(R_T),T_,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)    
    # Update temperature field
    T_melt = project(Tm)
    Tb_m = project(Tmb)
    Tb_temp = project(T_[N_T-1],Q)

    beta2.vector()[Tb_temp.vector()>(Tb_m.vector() - T_tol)] = 7e2 # EISMINT II 1e3 
    beta2.vector()[Tb_temp.vector()<=(Tb_m.vector() - T_tol)] = 1e9

    T_.vector()[T_.vector()>T_melt.vector()] = T_melt.vector()[T_.vector()>T_melt.vector()]

    # UPDATE PREVIOUS TIME STEP
    H0.interpolate(H)
    T0_.interpolate(T_)
    print t, H.vector().max()
    t+=dt

    # OUTPUT DATA
    Us_temp = project(as_vector([u(0.),v(0.0)]))
    Us.interpolate(Us_temp)
    w_temp = project(w(0),Q)
    ww.interpolate(w_temp)

    Ts_temp = project(T_[0],Q)
    Tb_temp = project(T_[N_T-1],Q)
    Ts_.interpolate(Ts_temp)
    Tb_.interpolate(Tb_temp)
    S_.interpolate(project(S,Q))

    
    Ufile << (Us,t)
    wfile << (ww,t)
    Hfile << (H,t)
    Sfile << (S_,t)

    Tbfile << (Tb_,t)
    Tsfile << (Ts_,t)
    Hxfile << H
    Txfile << T_

