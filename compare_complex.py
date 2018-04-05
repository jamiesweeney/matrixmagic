'''
    Jamie Sweeney
    April 2018

    Some more complex examples

'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math



# Example 1
#-- Take RGB image convert to greyscale
RGB_BINS = [0.299, 0.587, 0.114]

# Iterative version of turning to greyscale
def gsIterative(img):

    gs_img = np.zeros((img.shape[0],img.shape[1]))
    j = 0
    for y in img[:]:
        i = 0
        for x in y[:]:
            gs_val = 0
            k = 0
            for v in x[:]:
                gs_val = gs_val + v*RGB_BINS[k]
                k += 1
            gs_img[j,i] = gs_val
            i += 1
        j += 1

    return gs_img

# Vecor version
def gsVector(img):

    gs_img = np.zeros((img.shape[0],img.shape[1]))

    gs_img = img*RGB_BINS   # applies co-eff to each RGB val
    f_img = gs_img[:,:,0] + gs_img[:,:,1] + gs_img[:,:,2] # sum the spliced matrices

    return f_img

# Compares iterative and vector versions for (y,x) sized random RGB image
def compareGS(y, x, print_b):

    img = np.random.rand(y,x,3) * 255

    # Run iterative
    start = time.time()
    res_iter = gsIterative(img)
    t_iter = time.time() - start

    # Run vector
    start = time.time()
    res_vec = gsVector(img)
    t_vec = time.time() - start

    # Give output
    if (print_b == True):
        print ("Iterative : %.3f seconds." % t_iter)
        print ("Vector : %.3f seconds." % t_vec)
        print (np.array_equiv(res_iter, res_vec))

    return (t_iter, t_vec)



# Compares iterative and vector versions for all shapes of random RGB image
def compareGSMultiple(shapes, print_b):
    # Arrays for holding results, and axis
    r1 = np.array(()).reshape(0,1)
    r2 = np.array(()).reshape(0,1)
    s_axis = np.array(()).reshape(0,1)

    # Collect results for all shapes
    for shape in shapes:
        res = compareGS(shape[0], shape[1], print_b)

        r1 = np.vstack([r1, res[0]])
        r2 = np.vstack([r2, res[1]])

        # Adds log2(no. of elements)
        n = 1
        for s in shape:
            n = n*s
        s_axis = np.vstack([s_axis, math.log2(n)])

    # Plot the results
    plt.plot(s_axis, r1, label='Iterative')
    plt.plot(s_axis, r2, label='Vector')
    plt.ylabel('Time (s)')
    plt.xlabel('log2(size)')
    plt.legend()
    plt.show()

# Produces a set of matrix shapes
shapes = (())
num = 1
while num < 5000:
    shapes = shapes + ((num, num),)
    num = num*2

compareGSMultiple(shapes,True)
