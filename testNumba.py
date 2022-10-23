from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   
  
# normal function to run on cpu
def func(a):                                
    for i in range(10000000):
        a[i]+= 1      
  


if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
      
    funcjit = jit() (func)

    start = timer()
    func(a)
    print("without GPU:", timer()-start)    

    start = timer()
    funcjit(a)
    print("without GPU:", timer()-start)    