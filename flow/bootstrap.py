from calculations import *
import numpy as np


def bootstrap(estimator, size, m, *args):
    """General bootstap method."""

    def _inner(estimator, m, *args):
        index = np.random.randint(len(args[0]), size=len(args[0]))
        return estimator(m, *[x[index, ...] for x in args])

    return (_inner(estimator, m, *args) for _ in range(size))


class Bootstrap:

    def boot_vnm(self, b, m):
        """Bootstap v_n\{m\}."""
        cumulant = {2, 4}
        if m not in cumulant:
            raise ValueError("boot_vnm: m must be one of %r-cumulants." % cumulant)

        _ref = np.nansum(self.reference, 2)
        _dif = np.nansum(self.differential, 2)

        vnm = calc_vnm(m, _ref, _dif)  # The estimate
        boot = list(bootstrap(calc_vnm, b, m, _ref, _dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)

        if m == 2:  # 2-particle cumulant
            self.vn2['val'] = np.array(vnm)
            self.vn2['err'] = np.array(bs)
        else:  # 4-particle cumulant
            self.vn4['val'] = np.array(vnm)
            self.vn4['err'] = np.array(bs)

        return

    def boot_vnm_3sub_opposite(self, b, m, n_max):
        """Bootstap v_n\{m\}, where n <= n_max."""

        _ref = np.nansum(self.reference, 2)
        _dif = np.nansum(self.differential, 2)

        if m == 2:  # 2-particle cumulant
            vn2 = calc_vn2_3sub_opposite(_ref, _dif)  # The estimate
            boot = list(bootstrap(calc_vn2_3sub_opposite, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)

            self.vn2['val'] = np.array(vn2)
            self.vn2['err'] = np.array(bs)
        if m == 4:  # 4-particle cumulant
            vn4 = calc_vn4(_ref, _dif)  # The estimate

            boot = list(bootstrap(calc_vn4, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
            self.vn4['val'] = np.array(vn4)
            self.vn4['err'] = np.array(bs)

        return

    def boot_vnm_3sub_cms(self, b, m, n_max):
        """Bootstap v_n\{m\}, where n <= n_max."""

        _ref = np.nansum(self.reference, 2)
        _dif = np.nansum(self.differential, 2)

        if m == 2:  # 2-particle cumulant
            vn2 = calc_vn2_3sub_cms(_ref, _dif)  # The estimate
            boot = list(bootstrap(calc_vn2_3sub_cms, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)

            self.vn2['val'] = np.array(vn2)
            self.vn2['err'] = np.array(bs)
        if m == 4:  # 4-particle cumulant
            vn4 = calc_vn4(_ref, _dif)  # The estimate

            boot = list(bootstrap(calc_vn4, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
            self.vn4['val'] = np.array(vn4)
            self.vn4['err'] = np.array(bs)

        return

    def corr_vnm_ratio_3sub_opposite(self, ref, dif, ref1, dif1, b, m, n_max):
        _ref = np.nansum(ref, 2)
        _dif = np.nansum(dif, 2)
        _ref1 = np.nansum(ref1, 2)
        _dif1 = np.nansum(dif1, 2)
        if m == 2:
            vn = ratio_vn2(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn2, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        if m == 4:
            vn = ratio_vn4(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn4, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        bm = np.mean(boot, axis=0)
        bs = np.std(boot, axis=0)

        return np.array(vn), np.array(bs)

    def boot_vnmA(self, b, m, n_max):
        """Bootstap v_n\{m\}, where n <= n_max."""

        _ref = np.nansum(self.reference, 2)
        _dif = np.nansum(self.differential, 2)

        if m == 2:  # 2-particle cumulant
            vn2 = calc_vn2A(_ref, _dif)  # The estimate
            boot = list(bootstrap(calc_vn2A, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)

            self.vn2A['val'] = np.array(vn2)
            self.vn2A['err'] = np.array(bs)
        # if (m == 4): # 4-particle cumulant
        #     vn4=calc_vn4(_ref,_dif) # The estimate

        #     boot=list(bootstrap(calc_vn4,b,_ref, _dif)) # Get the bootstrap estimates
        #     bm,bs=np.mean(boot,axis=0), np.std(boot,axis=0)
        #     self.vn4['val'] = np.array(vn4)
        #     self.vn4['err'] = np.array(bs)

        return

    def boot_vnm_ref(self, b, m, n_max):
        """Bootstap v_n\{m\}, where n <= n_max."""

        _ref = np.nansum(self.reference, 2)
        _dif = np.nansum(self.differential, 2)

        if m == 2:  # 2-particle cumulant
            vn2 = calc_vn2_ref(_ref, _dif)  # The estimate
            boot = list(bootstrap(calc_vn2_ref, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)

            self.vn2_ref['val'] = np.array(vn2[:, :, :])
            self.vn2_ref['err'] = np.array(bs[:, :, :])
        if m == 4:  # 4-particle cumulant
            vn4 = calc_vn4_ref(_ref, _dif)  # The estimate

            boot = list(bootstrap(calc_vn4_ref, b, _ref, _dif))  # Get the bootstrap estimates
            bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
            self.vn4_ref['val'] = np.array(vn4[:, :, :])
            self.vn4_ref['err'] = np.array(bs[:, :, :])

        return

    def boot_vn2vtx(self, b, n_max):
        ref = self.reference
        dif = self.differential

        vn2 = calc_vn2_vtx(ref, dif)  # The estimate
        boot = list(bootstrap(calc_vn2_vtx, b, ref, dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
        self.vn2_vtx['val'] = np.array(vn2)
        self.vn2_vtx['err'] = np.array(bs)
        return

    def corr_vnm_ratio(self, ref, dif, ref1, dif1, b, m, n_max):
        _ref = np.nansum(ref, 2)
        _dif = np.nansum(dif, 2)
        _ref1 = np.nansum(ref1, 2)
        _dif1 = np.nansum(dif1, 2)
        if m == 2:
            vn = ratio_vn2(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn2, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        if m == 4:
            vn = ratio_vn4(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn4, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        bm = np.mean(boot, axis=0)
        bs = np.std(boot, axis=0)

        return np.array(vn), np.array(bs)

    def corr_vnm_3sub_opposite_ratio(self, ref, dif, ref1, dif1, b, m, n_max):
        _ref = np.nansum(ref, 2)
        _dif = np.nansum(dif, 2)
        _ref1 = np.nansum(ref1, 2)
        _dif1 = np.nansum(dif1, 2)
        if m == 2:
            vn = ratio_vnm_3sub_opposite(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vnm_3sub_opposite, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        if m == 4:
            vn = ratio_vn4(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn4, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        bm = np.mean(boot, axis=0)
        bs = np.std(boot, axis=0)

        return np.array(vn), np.array(bs)

    def corr_vnm_ratio_vtx(self, ref, dif, ref1, dif1, b, m, n_max):
        _ref = ref
        _dif = dif
        _ref1 = ref1
        _dif1 = dif1
        if m == 2:
            vn = ratio_vn2(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn2_vtx, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        if m == 4:
            vn = ratio_vn4(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn4_vtx, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        bm = np.mean(boot, axis=0)
        bs = np.std(boot, axis=0)

        return np.array(vn), np.array(bs)

    def corr_vnm_ref_ratio(self, ref, dif, ref1, dif1, b, m, n_max):
        _ref = np.nansum(ref, 2)
        _dif = np.nansum(dif, 2)
        _ref1 = np.nansum(ref1, 2)
        _dif1 = np.nansum(dif1, 2)
        if m == 2:
            vn = ratio_vn2_ref(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn2_ref, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        if m == 4:
            vn = ratio_vn4_ref(_ref1, _dif1, _ref, _dif)  # The estimate
            boot = list(bootstrap(ratio_vn4_ref, b, _ref, _dif, _ref1, _dif1))  # Get the bootstrap estimates
        bm = np.mean(boot, axis=0)
        bs = np.std(boot, axis=0)

        return np.array(vn[:, 0, :]), np.array(bs[:, 0, :])

    def boot_rnn(self, b, n_max):
        ref = np.nansum(self.reference, 2)
        dif = np.nansum(self.differential, 2)

        rnn = calc_rnn(ref, dif)  # The estimate
        boot = list(bootstrap(calc_rnn, b, ref, dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
        self.rnn['val'] = np.array(rnn)
        self.rnn['err'] = np.array(bs)

        return

    def boot_rnn_pPb(self, b, n_max):
        ref = np.nansum(self.reference, 2)
        dif = np.nansum(self.differential, 2)

        rnn = calc_rnn_pPb(ref, dif)  # The estimate
        boot = list(bootstrap(calc_rnn_pPb, b, ref, dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
        self.rnn['val'] = np.array(rnn)
        self.rnn['err'] = np.array(bs)

        return

    def boot_rnn_forward(self, b, n_max):
        ref = np.nansum(self.reference, 2)
        dif = np.nansum(self.differential, 2)

        rnn = calc_rnn_forward(ref, dif)  # The estimate
        boot = list(bootstrap(calc_rnn_forward, b, ref, dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
        self.rnn_forward['val'] = np.array(rnn)
        self.rnn_forward['err'] = np.array(bs)

        return

    def boot_rnn_backward(self, b, n_max):
        ref = np.nansum(self.reference, 2)
        dif = np.nansum(self.differential, 2)

        rnn = calc_rnn_backward(ref, dif)  # The estimate
        boot = list(bootstrap(calc_rnn_backward, b, ref, dif))  # Get the bootstrap estimates
        bm, bs = np.mean(boot, axis=0), np.std(boot, axis=0)
        self.rnn_backward['val'] = np.array(rnn)
        self.rnn_backward['err'] = np.array(bs)

        return
