import numpy as np


def true_with_probability(prob):
    # Check legal input or not
    if prob < 0 or prob > 1:
        raise ValueError('The argument prob must be in range [0, 1], got {:f}'.format(prob))
    
    return np.random.uniform(0, 1, 1) < prob


def bounded_normal(mean, std, lower_bound, upper_bound):
    # Sample from the normal distribution
    sample = np.random.normal(mean, std)
    
    # Bound the value into the specified range
    if sample < lower_bound:
        sample = lower_bound
    elif sample > upper_bound:
        sample = upper_bound
    
    return sample
