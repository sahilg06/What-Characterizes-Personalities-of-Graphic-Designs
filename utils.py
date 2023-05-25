'''
What Characterizes Personalities of Graphic Designs? SIGGRAPH 2018
Utils
'''

from keras import backend as K

def distance(vects):
    x, y = vects
    return x-y


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
