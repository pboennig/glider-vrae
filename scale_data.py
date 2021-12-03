from constants import *

def scale_data(X):
    return X * kScaleConstant

def unscale_data(X_scaled):
    return X_scaled / kScaleConstant
