##########################################################################################
# import necessary modules (list of all simulation running modules)
##########################################################################################
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mshr import *
from scipy import interpolate
#from ufl import *
##########################################################################################


##########################################################################################
# compliler settings / optimization options 
##########################################################################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 1
##########################################################################################


def generate_dataset(data):
    # mesh geometry  
    p_1_x = 0; p_1_y = 0;
    p_2_x = 28.0; p_2_y = 28.0;
    mesh = RectangleMesh(Point(p_1_x,p_1_y), Point(p_2_x,p_2_y), 28*5, 28*5, "right/left")
    # mesh and material prop
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2
    W = FunctionSpace(mesh, TH)
    V = FunctionSpace(mesh, 'CG', 1)
    back = 1.0
    high = 25.0
    nu = 0.3
    material_parameters = {'back':back, 'high':high, 'nu':nu}

    def bitmap(x,y): #there could be a much better way to do this, but this is working within the confines of ufl
        total = 0
        for j in range(0,data.shape[0]):
            for k in range(0,data.shape[1]):
                const1 = conditional(x>=j,1,0) # x is rows
                const2 = conditional(x<j+1,1,0)
                const3 = conditional(y>=k,1,0) # y is columns 
                const4 = conditional(y<k+1,1,0) #less than or equal to? 
                sum = const1 + const2 + const3 + const4
                const = conditional(sum>3,1,0) #ufl equality is not working, would like to make it sum == 4 
                total += const*data[j,k]
        return total

    class GetMat:
        def __init__(self,material_parameters,mesh):
            mp = material_parameters
            self.mesh = mesh
            self.back = mp['back']
            self.high = mp['high']
            self.nu = mp['nu']
        def getFunctionMaterials(self, V):
            self.x = SpatialCoordinate(self.mesh)
            val = bitmap(self.x[0],self.x[1])
            E = val/255.0*(self.high-self.back) + self.back
            effectiveMdata = {'E':E, 'nu':self.nu}
            return effectiveMdata

    mat = GetMat(material_parameters, mesh)
    EmatData = mat.getFunctionMaterials(V)
    E  = EmatData['E']
    nu = EmatData['nu']
    lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))
    matdomain = MeshFunction('size_t',mesh,mesh.topology().dim())
    dx = Measure('dx',domain=mesh, subdomain_data=matdomain)
    # define boundary domains 
    btm  =  CompiledSubDomain("near(x[1], btmCoord)", btmCoord = p_1_y)
    btmBC = DirichletBC(W, Constant((0.0,0.0)), btm)
    # apply traction, and body forces (boundary conditions are within the solver b/c they update)
    T  = Constant((0.0, 0.0))  # Traction force on the boundary
    B  = Constant((0.0, 0.0))
    # define finite element problem
    u = Function(W)
    du = TrialFunction(W)
    v = TestFunction(W)
    ##########################################################################################

    ################  ~ * ~ * ~ * ~ |
    ################  ~ * ~ * ~ * ~ |	--> solver loop and post-processing functions
    ################  ~ * ~ * ~ * ~ |

    ##########################################################################################
    def problem_solve(applied_disp,u,du,v):
        # Updated boundary conditions 
        top  =  CompiledSubDomain("near(x[1], topCoord)", topCoord = p_2_y)
        topBC = DirichletBC(W, Constant((0.0,applied_disp)), top)
        bcs = [btmBC,topBC]

        # Kinematics
        d = len(u)
        I = Identity(d) # Identity tensor
        F = I + grad(u) # Deformation gradient
        F = variable(F)

        psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))
        f_int = derivative(psi*dx,u,v)
        f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds , u, v)
        Fboth = f_int - f_ext
        # Tangent
        dF = derivative(Fboth, u, du)
        solve(Fboth == 0, u, bcs, J=dF)

        return u, du, v

    def pix_centers(u):
        disps_all_x = np.zeros(28*28)
        disps_all_y = np.zeros(28*28)
        for k in range(0,28):
            for j in range(0,28):
                X = j + 0.5 # x is columns
                Y = k + 0.5 # y is rows
                U = u(X,Y)
                disps_all_x[k*28+j] = U[0]
                disps_all_y[k*28+j] = U[1]

        return disps_all_x, disps_all_y

    # --> run the loop
    disp_val = [7.0, 11.0, 14.0]

    for disp in disp_val:
        applied_disp = disp
        u, du, v = problem_solve(applied_disp,u,du,v)
        if disp > 14.0-1e-9:
            return pix_centers(u)
##########################################################################################
