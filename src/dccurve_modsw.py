# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
import BayHunter.dccurve_ext as dccurve


class SurfDisp_dccurve(object):
    """Forward modeling of dispersion curves based on QGpCoreWave (Geopsy).

    """

    def __init__(self, obsx, ref):

        if (np.diff(obsx)>0.).any():
            # periods array has to be strictly decreasing! Otherwise dccurve will not work properly.
            self.reversed = True
        else:
            self.reversed = False

        self.obsx = obsx
        self.ref = ref

        self.modes = 1 

        self.wavetype, self.veltype = self.get_surftags(ref)


    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_surftags(self, ref):
        if ref == 'rdispgr':
            return (1, 1)

        elif ref == 'ldispgr':
            return (0, 1)

        elif ref == 'rdispph':
            return (1, 0)

        elif ref == 'ldispph':
            return (0, 0)
        else:
            tagerror = "Reference is not available in dccurve. If you defined \
a user Target, assign the correct reference (target.ref) or update the \
forward modeling plugin with target.update_plugin(MyForwardClass()).\n \
* Your ref was: %s\nAvailable refs are: rdispgr, ldispgr, rdispph, ldispph\n \
(r=rayleigh, l=love, gr=group, ph=phase)" % ref
            raise ReferenceError(tagerror)


    def run_model(self, h, vp, vs, rho, **params):
        """ The forward model will be run with the parameters below.

        """

        modes = self.modes

        # size of this array has to be multiplied with number of modes
        # here, we just take the zeroth mode into account
        dispvel = np.zeros(len(self.obsx))  # result

        if self.reversed:
            periods = self.obsx[::-1]
        else:
            periods = self.obsx

        dccurve.get_disp_curve(modes, self.veltype,
                   h, vp, vs, rho, periods, dispvel, self.wavetype)

        if self.reversed:
            dispvel = dispvel[::-1]

        if (dispvel == 0.).all():
            return np.nan, np.nan
        else:
            return self.obsx, 1./dispvel

