from dolfin import *
from pylab import deg2rad,linspace,zeros,argmin,array
from numpy.polynomial.legendre import leggauss
import time

parameters['form_compiler']['quadrature_degree'] = 4
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = 'auto'
parameters['form_compiler']['precision'] = 30
parameters['form_compiler']['epsilon'] = 1e-30

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

set_log_level(CRITICAL)

#   CRITICAL  = 50, // errors that may lead to data corruption and suchlike
#   ERROR     = 40, // things that go boom
#   WARNING   = 30, // things that may go boom later
#   INFO      = 20, // information of general interest
#   PROGRESS  = 16, // what's happening (broadly)
#   TRACE     = 13, // what's happening (in detail)
#   DBG       = 10  // sundry

##########################################################
###############        CONSTANTS       ###################
##########################################################
# Physical 
spy = 60**2*24*365
rho = 911.
rho_w = 1024.0
g = 9.81
n = 3.0

# Simulation
thklim =  0.1
thkinit = 100.

dt_float = 0.1
dt = Constant(dt_float)
eps_reg = 1e-10

# Expressions
class Adot(Expression):
  def eval(self,values,x):
    values[0] = 8.7/(1 + 200*exp(0.05e-3*(x[0]-500.e3))) - 8.0

class Mu(Expression):
  def eval(self,values,x):
    values[0] = .95/(1. + 9.*exp(x[0]/40.e3-10)) + 0.05

class Step(Expression):
    def __init__(self,threshold=.5,high=1.,low=0.):
        self.threshold = threshold
        self.high      = high
        self.low       = low
    def eval(self,values,x):
        if x[0] < self.threshold:
            values[0] = self.high
        else:
            values[0] = self.low

#########################################################
#################      GEOMETRY     #####################
#########################################################

class UpperSurface(Expression):
    def __init__(self,H,B):
        self.H = H
        self.B = B
    def eval(self,values,x):
        Expression.__init__(self)
        B  = self.B(x[0],x[1])
        H  = self.H(x[0],x[1])
        if B>0 or H>=-rho_w/rho*B:
            values[0] =  B+H
        else:
            values[0] = H * (1 - rho / rho_w)

class LowerSurface(Expression):
    def __init__(self,H,B):
        Expression.__init__(self)
        self.H = H
        self.B = B
    def eval(self,values,x):
        B  = self.B(x[0],x[1])
        H  = self.H(x[0],x[1])
        if B>0 or H>=-rho_w/rho*B:
            values[0] = B
        else:
            values[0] = -H * rho / rho_w

class Bed(Expression):
  def eval(self,values,x):
    xslope = 800.e3
    x1   = 1000./(1 + 200.*exp(0.10e-3*(x[0]-350.e3)))
    x2   = 500. /(1 + 200.*exp(0.05e-3*(x[0]-xslope)))
    values[0] = x1-x2-450

# Requires a seperate expression to be evaluated on nodes, and not centers of elements
class GroundedExp(Expression):
    def __init__(self,H=None,B=None,element=None):
        self.H = H
        self.B = B
    def eval(self,values,x):
        B  = self.B(x[0],x[1])
        H  = self.H(x[0],x[1])
        if H > -rho_w / rho * B:
            values[0] = 1.
        else:
            values[0] = 0.

# Sub domains for grounded/floating ice based on above expression
class Grounded(SubDomain):
    def __init__(self,GroundedProjection):
        SubDomain.__init__(self)
        self._GP = GroundedProjection

    def inside(self,x,on_boundary):
        return True if self._GP(x[0],x[1]) > 0. else False

#########################################################
#################       MESH        #####################
#########################################################
mesh = Mesh("mesh_kees_geometry.xml")
boundaries = MeshFunction('size_t', mesh, 'mesh_boundary_indicators.xml')
normal = FacetNormal(mesh)
h = CellSize(mesh)
hmin = project(h,FunctionSpace(mesh,'CG',1)).vector().min()

# Interior numbering
GROUNDED = 0 
FLOATING = 1 
domains     = CellFunction ("size_t", mesh)
domains.set_all(FLOATING)      # Default will be floating points.

# Boundary numbering
DIVIDE       = 0
MARGIN       = 1
SHELF        = 2
NOT_ASSIGNED = 3

dx = Measure("dx")[domains]
ds = Measure("ds")[boundaries]
#########################################################
#################  FUNCTION SPACES  #####################
#########################################################
Q = FunctionSpace(mesh,"CG",1)
Qcg3 = FunctionSpace(mesh,"CG",3)
Qdg = FunctionSpace(mesh,"DG",0)
Q2 = MixedFunctionSpace([Q]*2)
V = MixedFunctionSpace([Q]*4)
#########################################################
#################  FUNCTIONS  ###########################
#########################################################
B = interpolate(Bed(),Q)
adot = interpolate(Adot(),Q)
mu = interpolate(Mu(),Q)

# VELOCITY 
U = Function(V)
dU = TrialFunction(V)
Phi = TestFunction(V)

u,u2,v,v2 = split(U)
phi,phi1,psi,psi1 = split(Phi)

un = Function(Q)
u2n = Function(Q)
vn = Function(Q)
v2n = Function(Q)
w = Function(Q)

UU = Function(Q2)

# THICKNESS
H = Function(Q)
H0 = Function(Q)
input_dir = './startup/'
#File(input_dir+"H2480_1433.xml") >> H0
#File(input_dir+"H2480_1433.xml") >> H
H0 = interpolate(Step(threshold=300.e3,high=thkinit,low=thklim),Q)
H = interpolate(Step(threshold=300.e3,high=thkinit,low=thklim),Q)
dH = TrialFunction(Q)
xsi = TestFunction(Q)

taub  = Function(Q)
taud  = Function(Q)
tauxx = Function(Q)
tauxy = Function(Q)
tauxxo = Function(Q)
tauxyo = Function(Q)

Sl = LowerSurface(H,B) # Lower surface
Su = UpperSurface(H,B) # Upper surface
Slo = Function(Q) # Outputs for lower and upper surface expressions
Suo = Function(Q)
S = interpolate(Su,Q)
Slp = interpolate(Sl,Q)

grounded_pts = GroundedExp(H=H,B=B,element=Qcg3.ufl_element()) # Grounded ice expression (vertex)
grounded_projection = project(grounded_pts,Qdg)

grounded   = Grounded(grounded_projection) # Grounded ice mask
grounded.mark(domains,GROUNDED)            # Mark grounded (and partially grounded)

########################################################
#################   MOMENTUM BALANCE   #################
########################################################

# Vertical evaluation functions
def u_v(z):
  return u + u2*(S-z)**(n+1)

def v_v(z):
  return v + v2*(S-z)**(n+1)

def w_v(z):
  return -((u.dx(0) + v.dx(1))*(z-Slp) + 1/(n+2)*(u2.dx(0) + v2.dx(1))*(H**(n+2) - (S-z)**(n+2)) + (u2*S.dx(0) + v2*S.dx(1))*(H**(n+1) - (S-z)**(n+1)))

def dudx_v(z):
  return u.dx(0) + u2.dx(0)*(S-z)**(n+1) + (n+1)*u2*(S-z)**n*S.dx(0)

def dudy_v(z):
  return u.dx(1) + u2.dx(1)*(S-z)**(n+1) + (n+1)*u2*(S-z)**n*S.dx(1)

def dvdx_v(z):
  return v.dx(0) + v2.dx(0)*(S-z)**(n+1) + (n+1)*v2*(S-z)**n*S.dx(0)

def dvdy_v(z):
  return v.dx(1) + v2.dx(1)*(S-z)**(n+1) + (n+1)*v2*(S-z)**n*S.dx(1)

def dudz_v(z):
  return -(n+1)*u2*(S-z)**n

def dvdz_v(z):
  return -(n+1)*v2*(S-z)**n

# Viscosity
def eta_v(z):
  return A_v()**(-1./n)/2.*(dudx_v(z)**2 + dvdy_v(z)**2 + dudx_v(z)*dvdy_v(z) + 0.25*(dudz_v(z)**2 + dvdz_v(z)**2 + (dudy_v(z) + dvdx_v(z))**2) + eps_reg)**((1.-n)/(2*n))

def A_v():
  return 300.e3**(-3.)   # B = 300 kPa/yr**(1/3), A = B**-3

# Integrals for numerical quadrature in the vertical
def gauss_integral(dfdc,z_gauss,w):
  z = (S-Slp)/2.*z_gauss + (S+Slp)/2.
  return (S-Slp)/2.*w*eta_v(z)*dfdc(z)

# Calculate Gaussian approximants to integral terms
k = 7
points,weights = leggauss(int(k))
int_dudx = sum([gauss_integral(dudx_v,z,w) for z,w in zip(points,weights)])
int_dvdx = sum([gauss_integral(dvdx_v,z,w) for z,w in zip(points,weights)])
int_dudy = sum([gauss_integral(dudy_v,z,w) for z,w in zip(points,weights)])
int_dvdy = sum([gauss_integral(dvdy_v,z,w) for z,w in zip(points,weights)])

# Basal sliding parameterization (see van der Veen)
Np = H + rho_w / rho * Sl 

As = 1.0   # sliding constant should be 1.0
p  = 1.3   # exponent on effective pressure 1.3 is what Kees is using
m  = 3.    # non-linearity in the sliding law (exponent of velocity)

ubmag = sqrt(u_v(Slp)**2+v_v(Slp)**2 + 1e-3)
taubx = (mu * As * Np**p * ubmag**(1./m)) * u_v(Slp) / ubmag * grounded_projection 
tauby = (mu * As * Np**p * ubmag**(1./m)) * v_v(Slp) / ubmag * grounded_projection

tau_b = sqrt(taubx**2 + tauby**2 + 1e-1)
MIN_TRACTION = 1.e-2
tau_bx = conditional(ge(tau_b,MIN_TRACTION),taubx,MIN_TRACTION * taubx/tau_b)
tau_by = conditional(ge(tau_b,MIN_TRACTION),tauby,MIN_TRACTION * tauby/tau_b)

#tau_bx = conditional(ge(tau_bx,MIN_TRACTION),tau_bx,MIN_TRACTION)
#tau_by = conditional(ge(tau_by,MIN_TRACTION),tau_by,MIN_TRACTION)


# Driving stresses
tau_dx = rho*g*H*S.dx(0)
tau_dy = rho*g*H*S.dx(1)

# Grounded Ice Interior equations
R_field_x = - phi.dx(0)*(4*int_dudx + 2*int_dvdy) - phi.dx(1)*(int_dudy + int_dvdx) - phi*tau_bx - phi*tau_dx
R_field_y = - psi.dx(0)*(int_dudy + int_dvdx) - psi.dx(1)*(2*int_dudx + 4*int_dvdy) - psi*tau_by - psi*tau_dy

#  Grounded Ice Basal terms
R_shear_x = (eta_v(B)*(4*dudx_v(B) + 2*dvdy_v(B))*B.dx(0) + eta_v(B)*(dudy_v(B) + dvdx_v(B))*B.dx(1) - eta_v(B)*dudz_v(B) + tau_bx) * phi1
R_shear_y = (eta_v(B)*(dudy_v(B) + dvdx_v(B))*B.dx(0) + eta_v(B)*(4*dvdy_v(B) + 2*dudx_v(B))*B.dx(1) - eta_v(B)*dvdz_v(B) + tau_by) * psi1

# Ice shelf interior equations
F_field_x = - phi.dx(0)*(4*int_dudx + 2*int_dvdy) - phi.dx(1)*(int_dudy + int_dvdx) - phi*tau_dx
F_field_y = - psi.dx(0)*(int_dudy + int_dvdx) - psi.dx(1)*(2*int_dudx + 4*int_dvdy) - psi*tau_dy

# Ice shelf basal terms
F_shear_x = u2*phi1
F_shear_y = v2*psi1

# Ice shelf front
F_ocean_x = 1./2.*rho*g*(1-(rho/rho_w))*H**2*phi*normal[0]*ds(SHELF)
F_ocean_y = 1./2.*rho*g*(1-(rho/rho_w))*H**2*psi*normal[1]*ds(SHELF)

# Fjord walls and shear margin stress:
wall_stress = 1.e9 * rho * g * H
F_walls_x =  wall_stress*u_v(Slp)*phi*normal[1]*ds(MARGIN)
F_walls_y = -wall_stress*v_v(Slp)*psi*normal[0]*ds(MARGIN)

# L1L2
R = 0
R += (R_field_x + R_field_y + R_shear_x + R_shear_y)*dx(GROUNDED) 
R += (F_field_x + F_field_y + F_shear_x + F_shear_y)*dx(FLOATING) 
R += F_ocean_x + F_ocean_y
#R += F_walls_x + F_walls_y

J = derivative(R,U,dU)

#############################################################################
##########################  MASS BALANCE  ###################################
#############################################################################
ubar_ho = u + 1./(n+2)*u2*H0**(n+1)
vbar_ho = v + 1./(n+2)*v2*H0**(n+1)

umag_ho = sqrt(ubar_ho**2 + vbar_ho**2) 
kk = umag_ho*h  # This is some isotropic diffusion for stability 

# mass transport functional
# Other than artificial diffusion, this is standard continuity equation discritization
R_thick = ((H-H0)/dt*xsi + kk*dot(grad(H),grad(xsi)) + xsi*(Dx(ubar_ho*H,0) + Dx(vbar_ho*H,1)) - adot*xsi)*dx(GROUNDED) \
        + ((H-H0)/dt*xsi + kk*dot(grad(H),grad(xsi)) + xsi*(Dx(ubar_ho*H,0) + Dx(vbar_ho*H,1)) - adot*xsi)*dx(FLOATING)
J_thick = derivative(R_thick,H,dH)

#####################################################################
#########################  I/O Functions  ###########################
#####################################################################

# For moving data between vector functions and scalar functions 
assigner_inv = FunctionAssigner([Q,Q,Q,Q],V)
vec_assigner = FunctionAssigner(Q2,[Q,Q])
scalar_assigner = FunctionAssigner(Q,Q)

results_dir = '/home/jessej/model_output/backstress_crude/'
Hfile = File(results_dir + 'H.pvd')
Sufile = File(results_dir + 'Su.pvd')
Slfile = File(results_dir + 'Sl.pvd')
Ufile = File(results_dir + 'U.pvd')
gfile = File(results_dir + 'g.pvd')

tauxxfile = File(results_dir + 'tauxx.pvd')
tauxyfile = File(results_dir + 'tauxy.pvd')
taudfile = File(results_dir + 'taud.pvd')
taubfile = File(results_dir + 'taub.pvd')

Hxml = File(results_dir + 'H.xml')
Uxml = File(results_dir + 'U.xml')

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

#Define variational solver for the momentum problem

# Zero velocity on boundary
zero_velocity = [Constant(0.)]*4
m_bc = DirichletBC(V,zero_velocity,boundaries,MARGIN)
d_bc = DirichletBC(V,zero_velocity,boundaries,DIVIDE)
momentum_problem = NonlinearVariationalProblem(R,U,J=J,bcs=[m_bc,d_bc],form_compiler_parameters=ffc_options)

momentum_solver = NonlinearVariationalSolver(momentum_problem)
momentum_solver.parameters['nonlinear_solver'] = 'newton'
momentum_solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
momentum_solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
momentum_solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
momentum_solver.parameters['newton_solver']['maximum_iterations'] = 75
momentum_solver.parameters['newton_solver']['error_on_nonconvergence'] = False
momentum_solver.parameters['newton_solver']['linear_solver'] = 'mumps'

#Define variational solver for the mass problem
mass_problem = NonlinearVariationalProblem(R_thick,H,J=J_thick,form_compiler_parameters=ffc_options)
mass_solver = NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'

mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-6
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-6
mass_solver.parameters['snes_solver']['maximum_iterations'] = 20
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
mass_solver.parameters['snes_solver']['report'] = False

l_thick_bound = project(Constant(thklim),Q)
u_thick_bound = project(Constant(1e4),Q)

######################################################################
#######################   TIME LOOP   ################################
######################################################################

# Time interval
t = 0.0
t_end = 10000.
# Output every out_t years
out_t = 200.

def adaptive_update(momentum_solver, mass_solver,t,dt,dt_float):
    time_start = time.clock()
    SOLVED_U = False
    relaxation_parameter = 0.7
    momentum_solver.parameters['newton_solver']['relaxation_parameter'] = relaxation_parameter
    while not SOLVED_U:
        if relaxation_parameter < 0.2:
            status_U = [666,666]
            break
        status_U = momentum_solver.solve()
        SOLVED_U = status_U[1]
        if not SOLVED_U:
            relaxation_parameter /= 1.43
            momentum_solver.parameters['newton_solver']['relaxation_parameter'] = relaxation_parameter
            if MPI.rank(mpi_comm_world())==0:
                print "****************************************************************"
                print "WARNING: Newton relaxation parameter lowered to: "+str(relaxation_parameter)
                print "****************************************************************"

    # Solve mass equations, lowering time step on failure:
    SOLVED_H = False
    while not SOLVED_H:
        if dt_float < 1.e-5:
            status_H = [666,666]
            break
        status_H = mass_solver.solve(l_thick_bound,u_thick_bound)
        SOLVED_H = status_H[1]
        if not SOLVED_H:
            dt_float /= 2.
            dt.assign(dt_float)
            if MPI.rank(mpi_comm_world())==0:
                print "****************************************************************"
                print "WARNING: Time step lowered to: "+str(dt_float)
                print "****************************************************************"

    time_end = time.clock()
    run_time = time_end - time_start
    if MPI.rank(mpi_comm_world())==0:
      print "++++++++++++++++++++++++++++++++++++++++++++"
      print "Current time               : "+str(t)
      print "Current CFL based time step: "+str(dt_float)
      print "Momentum Newton iterations : "+str(status_U[0])
      print "Mass Newton iterations     : "+str(status_H[0])
      print "Time for to solve both (s) : "+str(run_time)
      print "Time remaining est. (h)    : "+str((t_end-t)/dt_float * run_time/60./60)
 
    t+=dt_float

    if SOLVED_U and SOLVED_H:
        return True,t
    else:
        return False,t

while t<t_end:
    # Solve momentum and mass equations, lowering relaxation paramter and time step on failure:
    SOLVED,t = adaptive_update(momentum_solver,mass_solver,t,dt,dt_float)
 
    # Re-mark domains
    domains.set_all(FLOATING)        # Default will be floating points.
    grounded_projection.assign(project(grounded_pts,Qdg))
    grounded.mark(domains,GROUNDED)

    # Update upper and lower surface interpolation
    S.assign(interpolate(Su,Q))
    Slp.assign(interpolate(Sl,Q))

    # Set previous thickness field
    H0.assign(H)

    # Guess next time step based on CFL:
    dt_float = min(hmin /  (2*project(umag_ho,Q).vector().max()),25.)
    dt.assign(dt_float)
     
    # This takes a step smaller than dt_float to reach the desired output time interval
    # Also writes output in the event of a solver fail
    if out_t - t % out_t <= dt_float or not SOLVED:
        if t % out_t != 0 and SOLVED:
            dt_save = dt_float
            dt_float = out_t -  t % out_t
            dt.assign(dt_float)

            SOLVED,t = adaptive_update(momentum_solver,mass_solver,t,dt,dt_float)
            if not SOLVED:
                break
            else:
                if MPI.rank(mpi_comm_world())==0:
                  print "===================================="
                  print "Output written at time     : "+str(t)
                  print "===================================="
             
         
        assigner_inv.assign([un,u2n,vn,v2n],U)
        vec_assigner.assign(UU,[un,vn])
        scalar_assigner.assign(Suo,interpolate(Su,Q))
        scalar_assigner.assign(Slo,interpolate(Sl,Q))
        scalar_assigner.assign(taud,project(sqrt(tau_dx**2+tau_dy**2),Q))
        #tbx=project(taubx,Q)
        #tby=project(tauby,Q)
        #tb = project(sqrt(tbx**2+tby**2),Q)
        #scalar_assigner.assign(taub,tb)

        scalar_assigner.assign(tauxxo,project(-Dx(project(eta_v(Sl+H/2.)*H*(4*Dx(un,0)+2*Dx(vn,1)),Q),0),Q))
        scalar_assigner.assign(tauxyo,project(-Dx(project(eta_v(Sl+H/2.)*H*(Dx(un,1)  +  Dx(vn,0)),Q),1),Q))

        Hfile  << (H,t)
        Sufile << (Suo,t)
        Slfile << (Slo,t)
        Ufile  << (UU,t)
        gfile  << (domains,t)

        #taubfile  << (taub,t)
        taudfile  << (taud,t)
        tauxxfile  << (tauxxo,t)
        tauxyfile  << (tauxyo,t)
        if SOLVED:
            Hxml << H
            Uxml << U
        else:
            break
