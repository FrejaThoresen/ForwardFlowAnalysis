import bootstrap
import helpers.general as gen
import numpy as np
import plotting
import root_numpy as rnp
import systematics
from helpers.constants import *
from rootpy.io import root_open


class Flow(bootstrap.Bootstrap, plotting.Plot, systematics.Sys):
    def __init__(self, filename, directory, eta_bins, cent_bins, cent_max, vertex_bins, samples, n_tot):
        self.eta_bins = eta_bins
        self.eta_list = gen.make_list(-3.36, 4.8, self.eta_bins)
        self.cent_bins = cent_bins
        self.cent_max = cent_max
        self.cent_list = np.linspace(0, cent_max, cent_bins)
        self.vertex_bins = vertex_bins
        self.no_samples = samples
        self.filename = filename
        self.directory = directory
        self.no_samples = samples
        self.eta_gap = True

        self.cent_list = np.linspace(0, self.cent_max, self.cent_bins)
        self.n_tot = n_tot

        # dimensions (n, pt, vertex, eta, centrality)
        self.c2 = {'val': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins)),
                   'err': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins))}
        self.d2 = {'val': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins)),
                   'err': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins))}
        self.d2A = {'val': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins)),
                    'err': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins))}
        self.vn2 = {'val': np.zeros((self.n_tot, self.eta_bins, 9)), 'err': np.zeros((self.n_tot, 9))}
        self.vn2A = {'val': np.zeros((self.n_tot, self.eta_bins, 9)), 'err': np.zeros((self.n_tot, 9))}
        self.vn2_ref = {'val': np.zeros((self.n_tot, 34, 9)), 'err': np.zeros((self.n_tot, self.eta_bins, 34, 9))}
        self.vn4_ref = {'val': np.zeros((self.n_tot, 34, 9)), 'err': np.zeros((self.n_tot, self.eta_bins, 34, 9))}
        self.c4 = {'val': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins)),
                   'err': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins))}
        self.d4 = {'val': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins)),
                   'err': np.zeros((self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins))}
        self.vn4 = {'val': np.zeros((self.n_tot, self.eta_bins, 9)), 'err': np.zeros((self.n_tot, self.eta_bins, 9))}
        self.SC42 = {'val': np.zeros((self.eta_bins, 9)), 'err': np.zeros((self.eta_bins, 9))}
        self.SC32 = {'val': np.zeros((self.eta_bins, 9)), 'err': np.zeros((self.eta_bins, 9))}
        self.sc42 = {'val': np.zeros((self.eta_bins, 9)), 'err': np.zeros((self.eta_bins, 9))}
        self.sc32 = {'val': np.zeros((self.eta_bins, 9)), 'err': np.zeros((self.eta_bins, 9))}

        self.vn2_vtx = {'val': np.zeros((self.n_tot, 10, self.eta_bins, 9)),
                        'err': np.zeros((self.n_tot, 10, self.eta_bins, 9))}

        self.rnnA = {'val': np.zeros((self.n_tot, self.vertex_bins, 5, self.cent_bins)),
                     'err': np.zeros((self.n_tot, self.vertex_bins, 5, self.cent_bins))}
        self.rnnB = {'val': np.zeros((self.n_tot, self.vertex_bins, 5, self.cent_bins)),
                     'err': np.zeros((self.n_tot, self.vertex_bins, 5, self.cent_bins))}
        self.rnn = {'val': np.zeros((self.n_tot, 5, 9)), 'err': np.zeros((self.n_tot, 5, 9))}
        self.rnn_forward = {'val': np.zeros((self.n_tot, 13, 9)), 'err': np.zeros((self.n_tot, 13, 9))}
        self.rnn_backward = {'val': np.zeros((self.n_tot, 7, 9)), 'err': np.zeros((self.n_tot, 7, 9))}

        self.reference = np.zeros((self.no_samples, self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins, 4))
        self.differential = np.zeros((self.no_samples, self.n_tot, self.vertex_bins, self.eta_bins, self.cent_bins, 6))
        self.differential_mixed = np.zeros((self.no_samples, self.vertex_bins, self.eta_bins, self.cent_bins, 5))
        self.differential_mixedn = np.zeros((self.no_samples, self.n_tot, self.vertex_bins, 14, self.cent_bins, 2))

        # (vn2{2}, v3{2}, vn4{2}, vn2{4}), centrality
        self.sys = {'tpc_pbpb': np.zeros((4, 9)), 'tpc_xexe': np.zeros((4, 9)), 'fmd_pbpb': np.zeros((4, 34, 9)),
                    'fmd_xexe': np.zeros((4, 34, 9))}  # [n,c], where n=4 corresponds to 4-particle cum. w. n=2.
        self.fill_sys()

    def fill_sys(self):

        self.sys['tpc_pbpb'][0, ...] = np.array([3.1, 2.3, 1.9, 1.5, 1.4, 1.4, 1.4, 0, 0]) * 0.01
        self.sys['tpc_pbpb'][1, ...] = np.array([3.9, 3.0, 2.2, 2.0, 2.3, 2.1, 2.7, 0, 0]) * 0.01
        self.sys['tpc_pbpb'][2, ...] = np.array([6.4, 5.2, 5.9, 6.2, 3.5, 6.6, 0.0, 0, 0]) * 0.01
        self.sys['tpc_pbpb'][3, ...] = np.array([0.0, 4.8, 2.0, 1.8, 1.5, 1.5, 3.3, 0, 0]) * 0.01

        self.sys['tpc_xexe'][0, ...] = np.array([4.1, 3.0, 2.7, 2.5, 2.4, 2.7, 2.4, 0, 0]) * 0.01
        self.sys['tpc_xexe'][1, ...] = np.array([6.1, 2.8, 2.8, 3.4, 3.7, 6.0, 0, 0, 0]) * 0.01

        a_pos = np.array([0.011, 0.02, 0.021])
        b_pos = np.array([0.979, 0.943, 0.928])
        res_pos = np.zeros((3, 13))
        for n in range(0, 3):
            res_pos[n] = a_pos[n] * self.eta_list[21:] + b_pos[n]
        res_pos = np.fabs(1 - res_pos) * 100

        a_neg = np.array([-0.039, -0.058, -0.078])
        b_neg = np.array([0.925, 0.868, 0.808])
        res_neg = np.zeros((3, 7))
        for n in range(0, 3):
            res_neg[n] = a_neg[n] * self.eta_list[0:7] + b_neg[n]
        res_neg = np.fabs(1 - res_neg) * 100

        self.sys['fmd_pbpb'][0, ...] = np.array([22.1, 19.7, 18.0, 17.4, 17.5, 17.2, 20.0, 0, 0])
        self.sys['fmd_pbpb'][1, ...] = np.array([32.6, 21.0, 18.1, 20.9, 27.4, 31.5, 47.8, 0, 0])
        self.sys['fmd_pbpb'][2, ...] = np.array([74.6, 79.7, 58.9, 69.9, 85.6, 123.6, 0.00, 0, 0])
        self.sys['fmd_pbpb'][3, ...] = np.array([21.6, 18.1, 17.0, 16.6, 16.7, 16.9, 20.2, 0, 0])

        self.sys['fmd_xexe'][0, ...] = np.array([20.4, 19.8, 14.9, 14., 13.8, 14.5, 17.5, 0, 0])
        self.sys['fmd_xexe'][1, ...] = np.array([45.1, 36.8, 34.2, 36.6, 43.9, 48.5, 0., 0, 0])

        # eta < 0
        self.sys['fmd_pbpb'][0:3, 0:7, ...] += np.power(res_neg[0:3, :, None], 2)
        self.sys['fmd_pbpb'][3, 0:7, ...] += np.power(res_neg[0, :, None], 2)
        self.sys['fmd_xexe'][0:3, 0:7, ...] += np.power(res_neg[0:3, :, None], 2)
        self.sys['fmd_xexe'][3, 0:7, ...] += np.power(res_neg[0, :, None], 2)

        # eta > 0
        self.sys['fmd_pbpb'][0:3, 21:, ...] += np.power(res_pos[0:3, :, None], 2)
        self.sys['fmd_pbpb'][3, 21:, ...] += np.power(res_pos[0, :, None], 2)
        self.sys['fmd_xexe'][0:3, 21:, ...] += np.power(res_pos[0:3, :, None], 2)
        self.sys['fmd_xexe'][3, 21:, ...] += np.power(res_pos[0, :, None], 2)

        self.sys['fmd_pbpb'][0:3, ...] = np.sqrt(self.sys['fmd_pbpb'][0:3, ...]) * 0.01
        self.sys['fmd_pbpb'][3, ...] = np.sqrt(self.sys['fmd_pbpb'][3, ...]) * 0.01
        self.sys['fmd_pbpb'][:, 7:21, ...] = float('NaN')

        self.sys['fmd_xexe'][0:3, ...] = np.sqrt(self.sys['fmd_xexe'][0:3, ...]) * 0.01
        self.sys['fmd_xexe'][3, ...] = np.sqrt(self.sys['fmd_xexe'][3, ...]) * 0.01
        self.sys['fmd_xexe'][:, 7:21, ...] = float('NaN')

    def get_vnm(self, m):
        if m == 2:
            return self.vn2
        if m == 4:
            return self.vn4
        return

    def read_cumulant_m(self, m):
        """
        read m-particle cumulants
        """
        myfile = root_open(self.filename, 'read')
        awesome = myfile.Get(self.directory)
        if self.eta_gap:
            cumu_rW2 = np.repeat(rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW2').Get('cumu_rW2')),
                                 34, axis=2)
            cumu_rW2Two = np.repeat(
                rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW2Two').Get('cumu_rW2Two')), 34, axis=3)
        else:
            cumu_rW2 = np.zeros((10, 10, 34, 80))
            cumu_rW2Two = np.zeros((3, 10, 10, 34, 80))
            cumu_rW2_tmp = rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW2').Get('cumu_rW2'))
            cumu_rW2Two_tmp = rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW2Two').Get('cumu_rW2Two'))
            cumu_rW2[:, :, 0:9, ...] = cumu_rW2_tmp[:, :, 0, None, ...]
            cumu_rW2[:, :, 9:19, ...] = cumu_rW2_tmp[:, :, 1, None, ...]
            cumu_rW2[:, :, 19:, ...] = cumu_rW2_tmp[:, :, 2, None, ...]
            cumu_rW2Two[:, :, :, 0:9, ...] = cumu_rW2Two_tmp[:, :, :, 0, None, ...]
            cumu_rW2Two[:, :, :, 9:19, ...] = cumu_rW2Two_tmp[:, :, :, 1, None, ...]
            cumu_rW2Two[:, :, :, 19:, ...] = cumu_rW2Two_tmp[:, :, :, 2, None, ...]

        cumu_dW2B = rnp.hist2array(awesome.Get('cumulants').Get('standard').Get('dW2B').Get('cumu_dW2B'))
        cumu_dW2TwoB = rnp.hist2array(awesome.Get('cumulants').Get('standard').Get('dW2TwoB').Get('cumu_dW2TwoB'))
        self.reference[:, 0, ..., rW2] = cumu_rW2
        self.reference[:, 1, ..., rW2] = cumu_rW2
        self.reference[:, 2, ..., rW2] = cumu_rW2
        self.reference[:, 0, ..., rW2Two] = cumu_rW2Two[0, ...]
        self.reference[:, 1, ..., rW2Two] = cumu_rW2Two[1, ...]
        self.reference[:, 2, ..., rW2Two] = cumu_rW2Two[2, ...]
        self.differential[:, 0, ..., dW2B] = cumu_dW2B
        self.differential[:, 1, ..., dW2B] = cumu_dW2B
        self.differential[:, 2, ..., dW2B] = cumu_dW2B
        self.differential[:, 0, ..., dW2TwoB] = cumu_dW2TwoB[0, ...]
        self.differential[:, 1, ..., dW2TwoB] = cumu_dW2TwoB[1, ...]
        self.differential[:, 2, ..., dW2TwoB] = cumu_dW2TwoB[2, ...]

        if m == 4:
            cumu_rW4 = np.repeat(rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW4').Get('cumu_rW4')),
                                 34, axis=2)
            cumu_rW4Four = np.repeat(
                rnp.hist2array(awesome.Get('cumulants').Get('reference').Get('rW4Four').Get('cumu_rW4Four')), 34,
                axis=3)

            cumu_dW4 = rnp.hist2array(awesome.Get('cumulants').Get('standard').Get('dW4').Get('cumu_dW4'))
            cumu_dW4Four = rnp.hist2array(awesome.Get('cumulants').Get('standard').Get('dW4Four').Get('cumu_dW4Four'))
            self.reference[:, 0, ..., rW4] = cumu_rW4
            self.reference[:, 1, ..., rW4] = cumu_rW4
            self.reference[:, 2, ..., rW4] = cumu_rW4
            self.reference[:, 0, ..., rW4Four] = cumu_rW4Four[0, ...]
            self.reference[:, 1, ..., rW4Four] = cumu_rW4Four[1, ...]
            self.reference[:, 2, ..., rW4Four] = cumu_rW4Four[2, ...]
            self.differential[:, 0, ..., dW4] = cumu_dW4
            self.differential[:, 1, ..., dW4] = cumu_dW4
            self.differential[:, 2, ..., dW4] = cumu_dW4
            self.differential[:, 0, ..., dW4Four] = cumu_dW4Four[0, ...]
            self.differential[:, 1, ..., dW4Four] = cumu_dW4Four[1, ...]
            self.differential[:, 2, ..., dW4Four] = cumu_dW4Four[2, ...]

    def read_2cumulant_Aside(self):
        datafile = root_open(self.filename, 'read')
        directory = datafile.Get(self.directory)

        cumu_dW2A = rnp.hist2array(directory.Get('cumulants').Get('standard').Get('dW2A').Get('cumu_dW2A'))
        cumu_dW2TwoA = rnp.hist2array(directory.Get('cumulants').Get('standard').Get('dW2TwoA').Get('cumu_dW2TwoA'))

        self.differential[:, 0, ..., dW2A] = cumu_dW2A
        self.differential[:, 1, ..., dW2A] = cumu_dW2A
        self.differential[:, 2, ..., dW2A] = cumu_dW2A
        self.differential[:, 0, ..., dW2TwoA] = cumu_dW2TwoA[0]
        self.differential[:, 1, ..., dW2TwoA] = cumu_dW2TwoA[1]
        self.differential[:, 2, ..., dW2TwoA] = cumu_dW2TwoA[2]

    def read_decorr(self):
        myfile = root_open(self.filename, 'read')
        awesome = myfile.Get(self.directory)

        cumu_dW2TwoTwoD = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('dW2TwoTwoD').Get('cumu_dW2TwoTwoD'))
        cumu_dW2TwoTwoN = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('dW2TwoTwoN').Get('cumu_dW2TwoTwoN'))

        self.differential_mixedn[..., dW2TwoTwoD] = cumu_dW2TwoTwoD
        self.differential_mixedn[..., dW2TwoTwoN] = cumu_dW2TwoTwoN

    def read_mixed(self):
        myfile = root_open(self.filename, 'read')
        awesome = myfile.Get(self.directory)
        cumu_dW22TwoTwoD = rnp.hist2array(
            awesome.Get('cumulants').Get('mixed').Get('dW22TwoTwoD').Get('cumu_dW22TwoTwoD'))
        cumu_dW22TwoTwoN = rnp.hist2array(
            awesome.Get('cumulants').Get('mixed').Get('dW22TwoTwoN').Get('cumu_dW22TwoTwoN'))

        self.differential_mixed[..., dW22TwoTwoD] = cumu_dW22TwoTwoD
        self.differential_mixed[..., dW22TwoTwoN] = cumu_dW22TwoTwoN

        cumu_dW4FourTwo = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('dW4FourTwo').Get('cumu_dW4FourTwo'))
        cumu_dW4ThreeTwo = rnp.hist2array(
            awesome.Get('cumulants').Get('mixed').Get('dW4ThreeTwo').Get('cumu_dW4ThreeTwo'))

        self.differential_mixed[..., dW4FourTwo] = cumu_dW4FourTwo
        self.differential_mixed[..., dW4ThreeTwo] = cumu_dW4ThreeTwo

        cumu_wSC = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('wSC').Get('cumu_wSC'))
        self.differential_mixed[..., wSC] = cumu_wSC

        cumu_dW2TwoTwoD = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('dW2TwoTwoD').Get('cumu_dW2TwoTwoD'))
        cumu_dW2TwoTwoN = rnp.hist2array(awesome.Get('cumulants').Get('mixed').Get('dW2TwoTwoN').Get('cumu_dW2TwoTwoN'))

        self.differential_mixedn[:, 0, ..., dW2TwoTwoD] = cumu_dW2TwoTwoD[0]
        self.differential_mixedn[:, 1, ..., dW2TwoTwoD] = cumu_dW2TwoTwoD[1]
        self.differential_mixedn[:, 2, ..., dW2TwoTwoD] = cumu_dW2TwoTwoD[2]
        self.differential_mixedn[:, 0, ..., dW2TwoTwoN] = cumu_dW2TwoTwoN[0]
        self.differential_mixedn[:, 1, ..., dW2TwoTwoN] = cumu_dW2TwoTwoN[1]
        self.differential_mixedn[:, 2, ..., dW2TwoTwoN] = cumu_dW2TwoTwoN[2]
