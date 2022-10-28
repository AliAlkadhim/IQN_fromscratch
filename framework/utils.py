import numpy as np
import pandas as pd

def normalize_IQN(values, expected_input_range):
    expected_range= expected_input_range#tuple of 2 arrays, each of shape (N_features,)
    #
    expected_min, expected_max = expected_range
    scale_factor = expected_max - expected_min
    offset = expected_min
    scaled_values = (values - offset)/scale_factor 
    # scaled_values = (values - np.mean(values))/np.std(values)
    return scaled_values

def denormalize_IQN(normalized_values, expected_input_range):
    expected_range=expected_input_range
    expected_min, expected_max = expected_range
    scale_factor = expected_max - expected_min
    offset = expected_min
    return normalized_values  * scale_factor + offset