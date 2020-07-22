import numpy as np


def make_list(xmin, xmax, xlen):
    x_list = np.linspace(xmin, xmax, xlen + 1)
    x_corr = -(abs(x_list[2]) - abs(x_list[1])) / 2
    x_center = x_list[:] - x_corr
    x_list = x_center[1:xlen + 1]
    return x_list


def make_phi_list(x_min, x_max, x_len):
    x_list = np.linspace(x_min, x_max, x_len + 1)
    x_corr = -(abs(x_list[2]) - abs(x_list[1])) / 2
    x_center = x_list[:] - x_corr
    return x_center[0:x_len]


def make_cent_list(x_min, x_max, x_len):
    cent_list = np.linspace(x_min, x_max, x_len + 1)
    cent_list = cent_list - x_max / x_len / 2.0
    return cent_list[1:]


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean', 'std']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def weighted_average(means, stds, axis):
    weights = np.divide(1, np.float_power(stds, 2), where=stds > 0, out=np.full_like(stds, float('NaN')))
    sumweights = np.nansum(weights, axis=axis)
    wx = np.nansum(weights * means, axis=axis)
    return np.divide(wx, sumweights, where=sumweights > 0, out=np.full_like(sumweights, float('NaN')))


def weighted_std_of_mean(means, stds, axis):
    weights = np.divide(1, np.float_power(stds, 2), where=stds > 0, out=np.full_like(stds, float('NaN')))
    sumweights = np.nansum(weights, axis=axis)
    return np.divide(1, np.sqrt(sumweights, where=sumweights > 0), where=sumweights > 0,
                     out=np.full_like(sumweights, float('NaN')))


def divide_samples(a, b):
    c2 = np.divide(ref[..., rW2Two], ref[..., rW2], \
                   where=ref[..., rW2] > 0, out=np.full_like(ref[..., rW2], float('NaN')))


def weighted_std(means, stds, axis):
    weightedMean = weightedAverage(means, stds, axis)
    weights = np.divide(1, np.float_power(stds, 2))

    if axis == 0:
        upper = np.nansum(weights * np.float_power(means - weightedMean, 2), axis=axis)
        lower = np.nansum(weights, axis=axis)

    if axis == 1:
        upper = np.nansum(weights * np.float_power(means - weightedMean[:, None], 2), axis=axis)
        lower = np.nansum(weights, axis=axis)

    if axis == 2:
        upper = np.nansum(weights * np.float_power(means - weightedMean[:, :, None], 2), axis=axis)
    if axis == 3:
        upper = np.nansum(weights * np.float_power(means - weightedMean[:, :, :, None], 2), axis=axis)
    if axis == 4:
        upper = np.nansum(weights * np.float_power(means - weightedMean[:, :, :, :, None], 2), axis=axis)

    return np.sqrt(np.divide(upper, lower, where=np.fabs(lower) > 0, out=np.full_like(lower, float('NaN'))))


def weighted_rebin(ndarray, ndarray_err, new_shape):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)

    if ndarray_err.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray_err.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray_err.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray_err = ndarray_err.reshape(flattened)
    for i in range(len(new_shape)):
        # ndarray_new = weightedAverage(ndarray,ndarray_err,-1*(i+1))#getattr(ndarray, operation)
        # ndarray_err_new = weightedStdofMean(ndarray,ndarray_err,-1*(i+1))#getattr(ndarray, operation)
        ndarray_new = weightedAverage(ndarray, ndarray_err, -1 * (i + 1))  # getattr(ndarray, operation)
        ndarray_err_new = weightedStdofMean(ndarray, ndarray_err, -1 * (i + 1))  # getattr(ndarray, operation)
        ndarray = ndarray_new
        ndarray_err = ndarray_err_new
    return ndarray, ndarray_err


def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args)
    evList = ['np.nanmean(' for i in range(lenShape)] + \
             ['a.reshape('] + ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + [',%d)' % (i + 1) for i in range(lenShape)]

    return eval(''.join(evList))


def apply_symmetry(vn, sigma, etalist):
    vn_sym = np.zeros((3, 34, 9))
    sigma_sym = np.zeros((3, 34, 9))
    for n in range(0, 3):
        vn_sym[n] = np.full_like(vn[n], float('NaN'))
        sigma_sym[n] = np.full_like(vn[n], float('NaN'))
    for n in range(0, 3):
        for c in range(0, 9):
            for etabin in range(0, 34):
                eta = etalist[etabin]
                if abs(eta) < 3.4:
                    negetabin = 27 - etabin
                    negeta = etalist[negetabin]
                    weights = [1 / np.float_power(sigma[n, etabin, c], 2),
                               1 / np.float_power(sigma[n, negetabin, c], 2)]
                    vn_sym[n, etabin, c] = (weights[0] * vn[n, etabin, c] + weights[1] * vn[n, negetabin, c]) / (
                                weights[0] + weights[1])
                    sigma_sym[n, etabin, c] = 1 / np.sqrt(weights[0] + weights[1])
                    vn_sym[n, negetabin, c] = vn_sym[n, etabin, c]
                    sigma_sym[n, negetabin, c] = sigma_sym[n, etabin, c]

                else:
                    vn_sym[n, etabin, c] = vn[n, etabin, c]
                    sigma_sym[n, etabin, c] = sigma[n, etabin, c]
    return vn_sym, sigma_sym
