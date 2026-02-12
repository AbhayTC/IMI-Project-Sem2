# import module 
import pymatgen.core as pg 
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer 

# assign and display data 
lattice = pg.Lattice.cubic(4.2) 
print('LATTICE\n', lattice, '\n') 
structure = pg.Structure(lattice, ["Li", "Cl"], 
			 [[0, 0, 0], 
			  [0.5, 0.5, 0.5]]) 
print('Structure', '\n', structure) 

# Convert structure of the compound 
# to user defined formats 
structure.to(fmt="poscar") 
structure.to(filename="POSCAR") 
structure.to(filename="newfile.cif") # name of the new file
