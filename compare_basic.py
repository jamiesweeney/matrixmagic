import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math

#-- The operations that we want to evaluate and their lamda functions --#

def f_identity(x):
    return x
# f_identity = lambda x: x

def f_addition(x):
    return x + 10
# f_addition = lambda x: x+10

def f_subtraction(x):
    return x - 10
# f_subtraction = lambda x: x-10

def f_multiplication(x):
    return x * 10
# f_multiplication = lambda x: x*10

def f_division(x):
    return (x / 10)
# f_division = lambda x: x/10


#-- The methods that we want to evaluate --#

# Performs a simple matrix operation by flattening the matrix to a 1D array
# and iterating over the elements.
def doIterativeOperation(matrix, func):
    ans = np.copy(matrix)
    ans = ans.flatten()

    i = 0
    while i < ans.shape[0]:
        ans[i] = func(ans[i])
        i += 1
    return ans.reshape(matrix.shape)

# Performs a simple matrix operation recursively.
def doRecursiveOperation(elem, func):
    if (type(elem) is np.ndarray):
        ans = np.copy(elem)
        i = 0
        while i < ans.shape[0]:
            ans[i] = doRecursiveOperation(ans[i], func)
            i += 1
        return ans
    else:
        ans = elem
        return func(ans)

# Performs a simple matrix by using scaler operations on the matrix
def doScalerOperation(matrix, func):
    ans = np.copy(matrix)
    return func(ans)


#-- Comparison logic --#

# Compares for a random array and a function
def compare(shape, l_func):

    # Generate random int (as float) matrix
    matrix = np.random.randint(low=0, high=1000, size=shape)
    matrix = matrix.astype(float)

    # Run iterative
    start = time.time()
    res_iter = doIterativeOperation(matrix, l_func)
    t_iter = time.time() - start

    # Run recursive
    start = time.time()
    res_rec = doRecursiveOperation(matrix, l_func)
    t_rec = time.time() - start

    # Run scaler
    start = time.time()
    res_sca = doScalerOperation(matrix, l_func)
    t_sca = time.time() - start


    # Give output
    print ("Iterative : %.3f seconds." % t_iter)
    print ("Recursive : %.3f seconds." % t_rec)
    print ("Scaler : %.3f seconds." % t_sca)
    print (np.array_equiv(res_iter, res_rec))
    print (np.array_equiv(res_rec, res_sca))

    return (t_iter, t_rec, t_sca)

# Compares
def compareMultiple(shapes, func):

    # Arrays for holding results, and axis
    r1 = np.array(()).reshape(0,1)
    r2 = np.array(()).reshape(0,1)
    r3 = np.array(()).reshape(0,1)
    s_axis = np.array(()).reshape(0,1)

    # Collect results for all shapes
    for shape in shapes:
        res = compare(shape, func)
        print (res)
        r1 = np.vstack([r1, res[0]])
        r2 = np.vstack([r2, res[1]])
        r3 = np.vstack([r3, res[2]])

        # Adds log2(no. of elements)
        n = 1
        for s in shape:
            n = n*s
        s_axis = np.vstack([s_axis, math.log2(n)])

    plt.plot(s_axis, r1, label='Iterative')
    plt.plot(s_axis, r2, label='Recursive')
    plt.plot(s_axis, r3, label='Scaler')
    plt.ylabel('Time (s)')
    plt.xlabel('log2(size)')
    plt.legend()
    plt.show()



# Produces a set of matrix shapes
shapes = (())
num = 1
while num < 10000:
    shapes = shapes + ((num, num),)
    num = num*2

compareMultiple(shapes, f_multiplication)
