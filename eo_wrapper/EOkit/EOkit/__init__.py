__all__ = ["lib", "ffi"]

import os
from .ffi import ffi
import sys


# The EOkit folder is built when pip installed. This folder would normally
# be git ignored; however, when building on ReadTheDocs, I need the .so file
# and whatnot. 
if sys.platform != "win32":
    lib = ""    
else:
    lib = open(os.path.join(os.path.dirname(__file__), "native.so"))
    
del os
