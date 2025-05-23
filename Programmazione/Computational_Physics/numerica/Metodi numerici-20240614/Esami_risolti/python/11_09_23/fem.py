from IPython.display import clear_output
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

#### PRELIMINARIES ####

def install():
    ready = True
    try:
        import dolfin
    except ImportError:
        ready = False
        print("Installing fenics... this should take about 30-60 seconds.")
        os.system('wget "https://fem-on-colab.github.io/releases/fenics-install-real.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')

    try:
        import gmsh
    except ImportError:
        ready = False
        print("Installing gmsh... this should take about 30-60 seconds.")
        os.system('wget "https://fem-on-colab.github.io/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"')
    clear_output()
    
install()

from fenics import dx, ds, grad, inner
deriv = lambda v: v.dx(0)
    
    
#### FINITE ELEMENT SPACES ####
    
def FEspace(mesh, r):
    from fenics import FunctionSpace
    V = FunctionSpace(mesh, 'CG', r)
    clear_output()
    return V
    
def dof2fun(u, V):
    from fenics import Function
    uf = Function(V)
    uf.vector()[:] = u + 0.0
    return uf

def fun2dof(uf):
    return uf.vector()[:]

def interpolate(expression, V):
    c = dofs(V).T
    if(callable(expression)):
      values = expression(*c)
    else:
      values = expression
    if(isinstance(values, list)):
        values = np.stack([v + 0*q for v, q in zip(values, c)], axis = 1).reshape(-1)

        from fenics import VectorElement, FunctionSpace
        element = VectorElement(V.ufl_element().family(), V.mesh().ufl_cell(), V.ufl_element().degree())
        Vvect = FunctionSpace(V.mesh(), element)
        clear_output()
        return dof2fun(values, Vvect)
    else:
        return dof2fun(values, V)

def dofs(V):
    return V.tabulate_dof_coordinates()

def plot(obj, axis = None, colorbar = True, shrink = 0.7, levels = None, spaceticks = False, title = None, **kwargs):
    import dolfin
    try:
        axis_ = "off" if axis is None else axis
        if(isinstance(obj, dolfin.cpp.mesh.Mesh)):
            dolfin.common.plotting.plot(obj)
        elif(isinstance(obj, dolfin.function.functionspace.FunctionSpace)):
            dolfin.common.plotting.plot(obj.mesh())
            c = dofs(obj).T
            if(len(c)==1):
              plt.plot(c[0], 0*c[0], '.r', label = 'dofs')
            else:
              plt.plot(*c, '.r', label = 'dofs')
        else:
            if(levels is None):
                levels = np.unique(fun2dof(obj)).shape[0]
            if(obj.function_space().mesh().geometry().dim() != 1):
                c = dolfin.common.plotting.plot(obj, cmap = 'jet', **kwargs)
                if(colorbar):
                    cbar = plt.colorbar(c, shrink = shrink)
                    if(spaceticks):
                        cbar.set_ticks([round(tick, 2) for tick in np.linspace(cbar.vmin, cbar.vmax, 6)])
            else:
              xdofs = dofs(obj.function_space()).reshape(-1)
              a, b = xdofs.min(), xdofs.max()
              xplot = np.linspace(a, b, 1000)
              line1, = plt.plot(xplot, [obj(x) for x in xplot], **{key: kwargs[key] for key in kwargs if key != "marker"})
              if("marker" in kwargs.keys()):
                 plt.plot(xdofs, fun2dof(obj), '.', color = line1.get_color())
              axis_ = "on"
        plt.axis(axis_)
        if(not (title is None)):
          plt.title(title)
    except Exception as e:
        raise RuntimeError("First argument should be either a dolfin.cpp.mesh.Mesh or a structure containing the dof values of some function (in which case 'space' must be != None).\n\nThis error was caused by the following exception: %s." % str(e))
    
class DirichletBC(object):
    def __init__(self, where, value):
        self.where = where
        self.value = value   

    def apply(self, F, V):
        from fenics import DirichletBC as dBC
        where = lambda x, on: on and self.where(*x)
        bds = dBC(V, interpolate(self.value, V), where).get_boundary_values()
        if(len(F.shape) == 2):
            F = F.tolil()
            for j in bds.keys():
              F[j, :] = 0
              F[j, j] = 1
            return F.tocsr()

        else:
            for key in bds.keys():
              F[key] = bds[key]
            return F

def applyBCs(F, V, *dbcs):
    for dbc in dbcs:
        F = dbc.apply(F, V)
    return F

def assemble(F, V):
    from fenics import TrialFunction, TestFunction, assemble as assmb
    u, v = TrialFunction(V), TestFunction(V)
    
    try:
        L = F(u, v)
        A = assmb(L)
        clear_output()
        return csr_matrix(A.array())  
    except:
        f = F(v)
        rhs = assmb(f)
        clear_output()
        return rhs[:]  
    

#### GEOMETRIES, DOMAINS and MESHES ####

class Domain(object):
    def __init__(self, main, other, operation = None):
        """Combines two domains via the specified operation."""
        self.a, self.b, self.op = main, other, operation
        self.index = 0
        self.dim = max(main.dim, other.dim)
  
    def script(self, index = 1):
        """Writes a gmsh script describing the domain."""
        res, j = self.a.script(index)
        res0, j0 = self.b.script(j)
        self.index = j0
        if(self.op == "u"):
            res0 += "BooleanUnion{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "i"):
            res0 += "BooleanIntersection{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "d"):
            res0 += "BooleanDifference{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        return res+res0, j0+1

    def __add__(self, other):
        return Domain(self, other, "u")

    def __sub__(self, other):
        return Domain(self, other, "d")

    def __mul__(self, other):
        return Domain(self, other, "i")

    def entity(self):
        if self.dim==2:
            return "Surface"
        elif self.dim==3:
            return "Volume"

class Rectangle(Domain):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Rectangle(%d) = {%f, %f, 0.0, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p1[0]-self.p0[0],
                                                             self.p1[1]-self.p0[1]), index+1

class Box(Domain):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 3

    def script(self, index = 1):
        self.index = index
        return 'Box(%d) = {%f, %f, %f, %f, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p0[2],
                                                          self.p1[0]-self.p0[0],self.p1[1]-self.p0[1],self.p1[2]-self.p0[2]), index+1

class Circle(Domain):
    def __init__(self, p, r = 1):
        self.p = p
        self.r = r
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Disk(%d) = {%f, %f, 0.0, %f};\n' % (index,self.p[0], self.p[1], self.r), index+1

class Polygon(Domain):
    def __init__(self, *points):
        self.p = points
        if(np.linalg.norm(np.array(points[0])- np.array(points[-1]))>1e-15):
            raise RuntimeError("First and last point should coincide.")
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        res = ""
        self.index = index
        n = len(self.p)-1
        for p in self.p[:-1]:
            res += "Point(%d) = {%f, %f, 0.0};\n" % (self.index,p[0],p[1])
            self.index += 1
        base = self.index
        for i in range(n-1):
            res += "Line(%d) = {%d, %d};\n" % (self.index,base-n+i,base-n+1+i)
            self.index += 1
        res += "Line(%d) = {%d, %d};\n" % (self.index,base-1,base-n)
        self.index += 1
        res += "Line Loop(%d) = {" % self.index
        for i in range(n):
            res += "%d, " % (self.index-n+i)
        res = res[:-2] + "};\n"
        self.index += 1
        res += "Plane Surface(%d) = {%d};\n" % (self.index, self.index-1)
        return res, self.index+1

class Line(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

def generate_mesh(domain, stepsize, structured = False):
   
    if(isinstance(domain, Line)):
        from fenics import IntervalMesh
        return IntervalMesh(int((domain.b-domain.a)/stepsize), domain.a, domain.b)
    else:
        import dolfin
        if(structured and domain.dim!=2):
            raise RuntimeError("Structured meshes are only available for 2D geometries.")
        code = 'SetFactory("OpenCASCADE");\nMesh.CharacteristicLengthMin = %f;\nMesh.CharacteristicLengthMax = %f;\n' % (stepsize, stepsize)
        code += domain.script()[0]
        extra = "\nTransfinite %s {%d};" %  (domain.entity(), domain.index) if structured else ""
        code += '\nPhysical %s(%d) = {%d};%s\nMesh.MshFileVersion = 2.0;' % (domain.entity(), domain.index+1, domain.index, extra)

        idf = np.random.randint(100000)
        print(code, file = open('%d.geo' % idf, 'w'))
        os.system("gmsh -%d %d.geo" % (domain.dim, idf))
        clear_output(wait = True)
        os.system("dolfin-convert %d.msh %d.xml" % (idf, idf))
        clear_output(wait = True)
        mesh = dolfin.cpp.mesh.Mesh("%d.xml" % idf)
        os.remove("%d.msh" % idf)
        os.remove("%d.xml" % idf)
        try:
            os.remove("%d_physical_region.xml" % idf)
        except:
            None
        os.remove("%d.geo" % idf)
        return mesh


#### ERROR COMPUTATION, L2 NORMS, POST-PROCESSING ####

def L2norm(e, mass_matrix):
    return np.sqrt(e @ (mass_matrix @ e))

def L2error(uex, ufem, domain, hquad = 0.05, rquad = 4):
    from fenics import interpolate as interp
    Vnew = FEspace(generate_mesh(domain, stepsize = hquad),  rquad)
    M = assemble(lambda u, v: u*v*dx, Vnew)
    clear_output()
    c = dofs(Vnew).T
    err = uex(*c) - interp(ufem, Vnew).vector()
    return L2norm(err, M)