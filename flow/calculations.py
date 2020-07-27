import numpy as np
from helpers.constants import *


def calc_vnm_reference(m, ref):
    """Calculate vn\{2\}.

    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result is averaged over number of samples, to get the mean value.

    """
    cumulant = {2, 4}
    if m not in cumulant:
        raise ValueError("calc_vnm: m must be one of %r-cumulants." % cumulant)

    cn = calc_cnm(m, ref)

    if m == 2:
        vnm = np.sqrt(cn, where=cn > 0, out=np.full_like(cn, float('NaN')))
    else:
        vnm = np.float_power(-cn, 1.0 / 4.0, where=cn < 0, out=np.full_like(cn, float('NaN')))

    return merge_centralities_vn(vnm)


def calc_cnm(m, ref):
    """Calculate c_n{2} or c_n{4}"""
    cumulant = {2, 4}
    if m not in cumulant:
        raise ValueError("calc_cnm: m must be one of %r-cumulants." % cumulant)

    c2 = np.divide(ref[..., rW2Two], ref[..., rW2], where=ref[..., rW2] > 0,
                   out=np.full_like(ref[..., rW2], float('NaN')))
    if m == 4:
        cn = np.divide(ref[..., rW4Four], ref[..., rW4], out=np.zeros_like(ref[..., rW4]),
                       where=ref[..., rW4] > 0) - 2 * np.float_power(c2, 2)
    else:
        cn = c2
    return cn


def calc_dnm(m, ref, dif):
    """Calculate d_n{2} or d_n{4}"""
    cumulant = {2, 4}
    if m not in cumulant:
        raise ValueError("calc_dnm: m must be one of %r-cumulants." % cumulant)

    d2 = np.divide(dif[..., dW2TwoB], dif[..., dW2B], where=dif[..., dW2B] > 0,
                   out=np.full_like(dif[..., dW2B], float('NaN')))
    if m == 4:
        c2 = calc_cnm(2, ref)
        dn = np.divide(dif[..., dW4Four], dif[..., dW4], out=np.zeros_like(dif[..., dW4]),
                       where=dif[..., dW4] > 0) - 2 * d2 * c2
    else:
        dn = d2
    return dn


def calc_vnm(m, ref, dif):
    """Calculate vn\{2\}.
    
    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """
    cumulant = {2, 4}
    if m not in cumulant:
        raise ValueError("calc_vnm: m must be one of %r-cumulants." % cumulant)

    cn = calc_cnm(2, ref)
    dn = calc_dnm(2, ref, dif)

    if m == 2:
        vnm = np.divide(dn, np.sqrt(cn, where=cn > 0, out=np.full_like(cn, float('NaN'))),
                        where=cn > 0, out=np.full_like(cn, float('NaN')))
    else:
        vnm = np.divide(-dn, np.float_power(-cn, 3.0 / 4.0,
                                            where=cn < 0, out=np.full_like(cn, float('NaN'))), where=cn < 0,
                        out=np.full_like(cn, float('NaN')))
    return merge_centralities_vn(vnm)


def calc_vn2_3sub_opposite(m, ref, dif):
    """Calculate vn\{2\}.

    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_cnm(2, ref)
    d2_Bside = calc_dnm(2, ref, dif)
    d2_Aside = np.divide(dif[..., dW2TwoA], dif[..., dW2A],
                         where=dif[..., dW2A] > 0, out=np.full_like(dif[..., dW2A], float('NaN')))

    vn2 = np.sqrt(np.divide(d2_Aside * d2_Bside, c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))))

    return merge_centralities_vn(vn2)




def calc_vn2A(m, ref, dif):
    """Calculate vn\{2\}.

    The output is average over centrality in the following binning: [0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90].
    The end result in then averaged over number of samples, to get the mean value.

    """

    c2 = calc_cnm(2, ref)

    d2 = np.divide(dif[..., dW2TwoA], dif[..., dW2A],
                   where=dif[..., dW2A] > 0, out=np.full_like(dif[..., dW2A], float('NaN')))

    vn2 = np.divide(d2, np.sqrt(c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))),
                    where=c2 > 0, out=np.full_like(c2, float('NaN')))

    return merge_centralities_vn(vn2)


def calc_vn2_vtx(m, ref, dif):
    c2 = calc_cnm(2, ref)

    d2 = np.divide(dif[..., dW2TwoB], dif[..., dW2B],
                   where=dif[..., dW2B] > 0, out=np.full_like(dif[..., dW2B], float('NaN')))

    vn2 = np.divide(d2, np.sqrt(c2, where=c2 > 0, out=np.full_like(c2, float('NaN'))),
                    where=c2 > 0, out=np.full_like(c2, float('NaN')))
    vn2_av = np.zeros((10, 3, 10, 34, 9))
    vn2_av[..., 0] = np.nanmean(vn2[..., 0:5], axis=4)
    vn2_av[..., 1] = np.nanmean(vn2[..., 5:10], axis=4)
    for c in range(2, 9):
        vn2_av[..., c] = np.nanmean(vn2[..., ((c - 1) * 10):(c * 10)], axis=4)

    return np.nanmean(vn2_av, 0)


def ratio_vnm(m, ref1, dif1, ref, dif):
    vn_1 = calc_vnm(ref1, dif1, m)
    vn = calc_vnm(ref, dif, m)
    return np.divide(vn, vn_1)


def ratio_vnm_3sub_opposite(m, ref1, dif1, ref, dif):
    vn2_1 = calc_vn2_3sub_opposite(ref1, dif1)
    vn2 = calc_vn2_3sub_opposite(ref, dif)
    return np.divide(vn2, vn2_1)


def ratio_vnm_reference(m, ref1, ref):
    vnm_1 = calc_vnm_reference(ref1, m)
    vnm = calc_vnm_reference(ref, m)
    return np.divide(vnm, vnm_1)


def calc_rnn(m, ref, dif):
    rnnA = calc_rnn_Aside(ref, dif)
    rnnB = calc_rnn_Aside(ref, dif)

    return rnnA  # np.nanmean([rnnA,rnnB],axis=0)


def calc_rnn_pPb(m, ref, dif):
    rnn_Aside = calc_rnn_Aside(ref, dif)
    rnn_Bside = calc_rnn_Bside(ref, dif)

    return np.sqrt(rnn_Aside * rnn_Bside)  # np.nanmean([rnnA,rnnB],axis=0)


def calc_rnn_backward(m, ref, dif):
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
    result_av = np.zeros((10, 3, 34, 9))
    result_av[..., 0] = np.nanmean(result[..., 0:5], axis=3)
    result_av[..., 1] = np.nanmean(result[..., 5:10], axis=3)
    for c in range(2, 9):
        result_av[..., c] = np.nanmean(result[..., ((c - 1) * 10):(c * 10)], axis=3)
    return np.nanmean(result_av, axis=0)


def merge_centralities_vn(result):
    """
    pretty static function to merge results into centrality bins of [0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60]
    """
    result_av = np.zeros((10, 3, 34, 9))
    result_av[..., 0] = np.nanmean(result[..., 0:5], axis=3)
    result_av[..., 1] = np.nanmean(result[..., 5:10], axis=3)
    for c in range(2, 9):
        result_av[..., c] = np.nanmean(result[..., ((c - 1) * 10):(c * 10)], axis=3)
    return np.nanmean(result_av, axis=0)


def calc_rnn_Aside(m, ref, dif):
    numerator = np.divide(dif[:, :, 9:14, :, dW2TwoB], dif[:, :, 9:14, :, dW2B], where=dif[:, :, 9:14, :, dW2B] > 0,
                          out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    denominator = np.divide(dif[:, :, 14:19, :, dW2TwoA], dif[:, :, 14:19, :, dW2A],
                            where=dif[:, :, 14:19, :, dW2A] > 0,
                            out=np.full_like(dif[:, :, 14:19, :, dW2A], float('NaN')))
    rnn = np.divide(np.flip(numerator, 2), denominator, where=denominator != 0,
                    out=np.full_like(denominator, float('NaN')))

    return merge_centralities(rnn)


def calc_rnn_Bside(m, ref, dif):
    numerator = np.divide(dif[:, :, 9:14, :, dW2TwoA], dif[:, :, 9:14, :, dW2A], where=dif[:, :, 9:14, :, dW2A] > 0,
                          out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    denominator = np.divide(dif[:, :, 14:19, :, dW2TwoB], dif[:, :, 14:19, :, dW2B],
                            where=dif[:, :, 14:19, :, dW2B] > 0,
                            out=np.full_like(dif[:, :, 9:14, :, dW2B], float('NaN')))
    rnn = np.divide(1,
                    np.divide(np.flip(numerator, 2), denominator, where=denominator != 0,
                              out=np.full_like(denominator, float('NaN'))),
                    where=denominator != 0, out=np.full_like(denominator, float('NaN')))

    return merge_centralities(rnn)
