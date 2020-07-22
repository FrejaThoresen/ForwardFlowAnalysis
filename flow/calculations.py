import numpy as np
from helpers.constants import *


def calc_vn2(ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = np.divide(ref[..., rW2Two], ref[..., rW2], \
                   where=ref[..., rW2] > 0, out=np.full_like(ref[..., rW2], float('NaN')))
    d2 = np.divide(dif[..., dW2TwoB], dif[..., dW2B], \
                   where=dif[..., dW2B] > 0, out=np.full_like(dif[..., dW2B], float('NaN')))

    vn2 = np.divide(d2, np.sqrt(c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))), \
                    where=c2 > 0, out=np.full_like(c2, float('NaN')))

    vn2_av = np.zeros((10, 3, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=3)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=3)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=3)

    return np.nanmean(vn2_av, axis=0)


def calc_vn2_3sub_opposite(ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = np.divide(ref[..., rW2Two], ref[..., rW2], \
                   where=ref[..., rW2] > 0, out=np.full_like(ref[..., rW2], float('NaN')))
    d2B = np.divide(dif[..., dW2TwoB], dif[..., dW2B], \
                    where=dif[..., dW2B] > 0, out=np.full_like(dif[..., dW2B], float('NaN')))
    d2A = np.divide(dif[..., dW2TwoA], dif[..., dW2A], \
                    where=dif[..., dW2A] > 0, out=np.full_like(dif[..., dW2A], float('NaN')))

    vn2 = np.sqrt(np.divide(d2A * d2B, c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))))

    vn2_av = np.zeros((10, 3, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=3)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=3)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=3)

    return np.nanmean(vn2_av, axis=0)

def calc_c2(ref):
    return np.divide(ref[..., rW2Two], ref[..., rW2], where=ref[..., rW2] > 0, out=np.full_like(ref[..., rW2], float('NaN')))


def calc_vn2_3sub_cms(ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_c2(ref)
    d2B = np.divide(dif[..., dW2TwoB], dif[..., dW2B], \
                    where=dif[..., dW2B] > 0, out=np.full_like(dif[..., dW2B], float('NaN')))
    d2A = np.divide(dif[..., dW2TwoA], dif[..., dW2A], \
                    where=dif[..., dW2A] > 0, out=np.full_like(dif[..., dW2A], float('NaN')))

    under = np.divide(c2 * d2A, d2B, where=d2B > 0, out=np.full_like(d2B, float('NaN')))

    vn2 = d2A / np.sqrt(under, where=under > 0, out=np.full_like(under, float('NaN')))

    # vn2 = np.sqrt(np.divide(d2A*d2B,c2,where=c2>0,out=np.full_like(c2, float('NaN'))))

    vn2_av = np.zeros((10, 3, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=3)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=3)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=3)

    return np.nanmean(vn2_av, axis=0)


def calc_vn2A(ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_c2(ref)

    d2 = np.divide(dif[..., dW2TwoA], dif[..., dW2A],
                   where=dif[..., dW2A] > 0, out=np.full_like(dif[..., dW2A], float('NaN')))

    vn2 = np.divide(d2, np.sqrt(c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))), \
                    where=c2 > 0, out=np.full_like(c2, float('NaN')))

    vn2_av = np.zeros((10, 3, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=3)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=3)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=3)

    return np.nanmean(vn2_av, axis=0)


def calc_vn2_ref(ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """
    vn2 = np.sqrt(np.divide(ref[..., rW2Two], ref[..., rW2], \
                            where=ref[..., rW2] > 0, out=np.full_like(ref[..., rW2], float('NaN'))))

    return merge_centralities_vn(vn2)


def calc_vn2vtx(ref, dif):
    c2 = calc_c2(ref)

    d2 = np.divide(dif[..., dW2TwoB], dif[..., dW2B], \
                   where=dif[..., dW2B] > 0, out=np.full_like(dif[..., dW2B], float('NaN')))

    vn2 = np.divide(d2, np.sqrt(c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))), \
                    where=c2 > 0, out=np.full_like(c2, float('NaN')))
    vn2_av = np.zeros((10, 3, 10, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=4)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=4)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=4)

    return np.nanmean(vn2_av, 0)


def calc_vn4(ref, dif):
    """Calculate vn\{4\}.

    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_c2(ref)
    d2 = np.divide(dif[..., dW2TwoB], dif[..., dW2B], out=np.zeros_like(dif[..., dW2B]), where=dif[..., dW2B] > 0)
    c4 = np.divide(ref[..., rW4Four], ref[..., rW4], out=np.zeros_like(ref[..., rW4]),
                   where=ref[..., rW4] > 0) - 2 * np.float_power(c2, 2)
    d4 = np.divide(dif[..., dW4Four], dif[..., dW4], out=np.zeros_like(dif[..., dW4]),
                   where=dif[..., dW4] > 0) - 2 * d2 * c2
    vn4 = np.divide(-d4, np.float_power(-c4, 3.0 / 4.0,
                                        where=c4 < 0, out=np.full_like(c4, float('NaN'))), where=c4 < 0,
                    out=np.full_like(c4, float('NaN')))

    return merge_centralities_vn(vn4)


def calc_vn4_ref(ref, dif):
    """Calculate vn\{4\}.

    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_c2(ref)
    c4 = np.divide(ref[..., rW4Four], ref[..., rW4], out=np.zeros_like(ref[..., rW4]),
                   where=ref[..., rW4] > 0) - 2 * np.float_power(c2, 2)
    vn4 = np.float_power(-c4, 1.0 / 4.0, where=c4 < 0, out=np.full_like(c4, float('NaN')))

    vn4_av = np.zeros((10, 3, 34, 9))
    vn4_av[..., 0] = np.nanmean(vn4[..., 0:5], axis=3)
    vn4_av[..., 1] = np.nanmean(vn4[..., 5:10], axis=3)
    for c in range(2, 9):
        vn4_av[..., c] = np.nanmean(vn4[..., ((c - 1) * 10):(c * 10)], axis=3)
    return np.nanmean(vn4_av, 0)


def ratio_vn2(ref1, dif1, ref, dif):
    vn2_1 = calc_vn2(ref1, dif1)
    vn2 = calc_vn2(ref, dif)
    return np.divide(vn2, vn2_1)


def ratio_vnm_3sub_opposite(ref1, dif1, ref, dif):
    vn2_1 = calc_vn2_3sub_opposite(ref1, dif1)
    vn2 = calc_vn2_3sub_opposite(ref, dif)
    return np.divide(vn2, vn2_1)


def ratio_vn2_ref(ref1, dif1, ref, dif):
    vn2_1 = calc_vn2_ref(ref1, dif1)
    vn2 = calc_vn2_ref(ref, dif)
    return np.divide(vn2, vn2_1)


def ratio_vn4(ref1, dif1, ref, dif):
    vn4_1 = calc_vn4(ref1, dif1)
    vn4 = calc_vn4(ref, dif)
    return np.divide(vn4, vn4_1)


def ratio_vn4_ref(ref1, dif1, ref, dif):
    vn4_1 = calc_vn4_ref(ref1, dif1)
    vn4 = calc_vn4_ref(ref, dif)
    return np.divide(vn4, vn4_1)


def calc_rnn(ref, dif):
    rnnA = calc_rnnA(ref, dif)
    rnnB = calc_rnnA(ref, dif)

    return rnnA  # np.nanmean([rnnA,rnnB],axis=0)


def calc_rnn_pPb(ref, dif):
    rnnA = calc_rnnA(ref, dif)
    rnnB = calc_rnnB(ref, dif)

    return np.sqrt(rnnA * rnnB)  # np.nanmean([rnnA,rnnB],axis=0)


def calc_rnn_backward(ref, dif):
    N = np.divide(dif[:, :, 0:7, :, dW2TwoB], dif[:, :, 0:7, :, dW2B], where=dif[:, :, 0:7, :, dW2B] > 0,
                  out=np.full_like(dif[:, :, 0:7, :, dW2B], float('NaN')))
    D = np.divide(dif[:, :, 0:7, :, dW2TwoA], dif[:, :, 0:7, :, dW2A], where=dif[:, :, 0:7, :, dW2A] > 0,
                  out=np.full_like(dif[:, :, 0:7, :, dW2A], float('NaN')))
    rnn = np.divide(N, D, where=D != 0, out=np.full_like(D, float('NaN')))

    return merge_centralities(rnn)



def merge_centralities(result):
    """
    pretty static function to merge results into centrality bins of [0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60]
    """
    result_av = np.zeros((10, 3, 13, 9))
    result_av[..., 0] = np.nanmean(result[..., 0:5], axis=3)
    result_av[..., 1] = np.nanmean(result[..., 5:10], axis=3)
    for c in range(2, 9):
        result_av[..., c] = np.nanmean(result[..., ((c - 1) * 10):(c * 10)], axis=3)
    return np.nanmean(rnn_av, axis=0)


def merge_centralities_vn(result):
    """
    pretty static function to merge results into centrality bins of [0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60]
    """
    result_av = np.zeros((10, 3, 34, 9))
    result_av[..., 0] = np.nanmean(result[..., 0:5], axis=3)
    result_av[..., 1] = np.nanmean(result[..., 5:10], axis=3)
    for c in range(2, 9):
        result_av[..., c] = np.nanmean(result[..., ((c - 1) * 10):(c * 10)], axis=3)
    return np.nanmean(rnn_av, axis=0)


def calc_rnn_forward(ref, dif):
    numerator = np.divide(dif[:, :, 21:, :, dW2TwoB], dif[:, :, 21:, :, dW2B], where=dif[:, :, 21:, :, dW2B] > 0,
                          out=np.full_like(dif[:, :, 21:, :, dW2B], float('NaN')))
    denominator = np.divide(dif[:, :, 21:, :, dW2TwoA], dif[:, :, 21:, :, dW2A], where=dif[:, :, 21:, :, dW2A] > 0,
                            out=np.full_like(dif[:, :, 21:, :, dW2A], float('NaN')))
    rnn = np.divide(numerator, denominator, where=denominator != 0, out=np.full_like(denominator, float('NaN')))

    return merge_centralities(rnn)


def calc_rnnA(ref, dif):
    N = np.divide(dif[:, :, 9:14, :, dW2TwoB], dif[:, :, 9:14, :, dW2B], where=dif[:, :, 9:14, :, dW2B] > 0,
                  out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    D = np.divide(dif[:, :, 14:19, :, dW2TwoA], dif[:, :, 14:19, :, dW2A], where=dif[:, :, 14:19, :, dW2A] > 0,
                  out=np.full_like(dif[:, :, 14:19, :, dW2A], float('NaN')))
    rnn = np.divide(np.flip(N, 2), D, where=D != 0, out=np.full_like(D, float('NaN')))

    return merge_centralities(rnn)


def calc_rnnB(ref, dif):
    N = np.divide(dif[:, :, 9:14, :, dW2TwoA], dif[:, :, 9:14, :, dW2A], where=dif[:, :, 9:14, :, dW2A] > 0,
                  out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    D = np.divide(dif[:, :, 14:19, :, dW2TwoB], dif[:, :, 14:19, :, dW2B], where=dif[:, :, 14:19, :, dW2B] > 0,
                  out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    rnn = np.divide(1,
                    np.divide(np.flip(N, 2), D, where=D != 0, out=np.full_like(D, float('NaN'))),
                    where=D != 0, out=np.full_like(D, float('NaN')))

    return merge_centralities(rnn)

