# Insert your correction here
import numpy as np
# a vector as a Python list datatype
asList = [1,2,3]

# same numbers, but as a dimensionless numpy array
asArray = np.array([1,2,3])

# again same numbers, but now endowed with orientations
rowVec = np.array([ [1,2,3] ]) # row
colVec = np.array([ [1],[2],[3] ]) # column

# Note the use of spacing when defining the vectors; that is not necessary but makes the code more readable.