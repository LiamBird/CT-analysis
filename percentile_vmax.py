## Function to return max and min saturation values excluding saturation at high and low values:
def percentile_vmax(arrays, percentile=95, absolute=True, ignore_black=True, subsample_large=True, subsample_n=1000):
    import numpy as np
    arrays = np.hstack([array.flatten() for array in arrays])
    if subsample_large == True:
        if np.prod(arrays.shape) > 10e3:
            random_list = []
            max_value = np.prod(arrays.shape)
            while len(random_list) < subsample_n:
                new_value = np.random.randint(max_value-1)
                if new_value not in random_list:
                    random_list.append(new_value)

        arrays = arrays[random_list]                

    if ignore_black == True:
        arrays = arrays[np.nonzero(arrays)]
    if absolute == True:
        sorted_arr = np.sort(abs(arrays))
    else:
        sorted_arr = np.sort(arrays)
    return sorted_arr[int(percentile/100*sorted_arr.shape[0])]  
