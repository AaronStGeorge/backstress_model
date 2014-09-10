from dolfin import *
from numpy import linspace

#########################################################
#################      GEOMETRY     #####################
#########################################################
length = 800.e3
height = 50.e3
r = Rectangle(0.,0.,length,height)
mesh = Mesh(r,100) # 150 is about 2.5 km

REFINE_LEVEL = 0
# Refine in region of interest, where gl moves
for i in range(REFINE_LEVEL):
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in cells(mesh):
        xpos = cell.midpoint().x()
        if xpos > 200.e3 and xpos < 570.e3:
            cell_markers[cell] = True 
    mesh = refine(mesh, cell_markers)

mesh.smooth(100)
h = CellSize(mesh)

print "Some mesh data:"
print "hmin: " + str(mesh.hmin())
print "hmax: " + str(mesh.hmax())
print "vertices: " + str(mesh.num_vertices())

# SubDomains for Boundary Conditions
class Symmetry(SubDomain):
    def inside(self,x,on_boundary):
        return near(abs(x[1]), 0.) and on_boundary 

class FreeSlip(SubDomain):
    def inside(self,x,on_boundary):
        return near(abs(x[1]),height) and on_boundary 

class ShelfFront(SubDomain):
    def inside(self,x,on_boundary):
        return near(abs(x[0]),length) and on_boundary

class Divide(SubDomain):
    def inside(self,x,on_boundary):
        return near(abs(x[0]),0.) and on_boundary


# Numbering scheme
SYMMETRY       = 0
FREESLIP       = 1
SHELF          = 2
DIVIDE         = 3
NOT_ASSIGNED   = 4

boundaries  = FacetFunction("size_t", mesh)
# Mark them:
boundaries.set_all(NOT_ASSIGNED)   # Default will be anything but boundary.

# Think about order here, it matters

divide = Divide()
divide.mark(boundaries,DIVIDE)

freeslip = FreeSlip()
freeslip.mark(boundaries,FREESLIP)

symmetry = Symmetry()
symmetry.mark(boundaries,SYMMETRY)

shelf = ShelfFront()
shelf.mark(boundaries,SHELF)

plot(boundaries,interactive=True)

File("mismip3d_geometry.xml")<<mesh
File("mismip3d_boundary_indicators.xml")<<boundaries
