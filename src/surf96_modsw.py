# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
from BayHunter.surfdisp96_ext import surfdisp96, depthkernel


class SurfDisp(object):
    """Forward modeling of dispersion curves based on surf96 (Rob Herrmann).

    The forward calculation is based on PySurf96AA (https://github.com/ekaestle/pysurf96aa). Please see the references therein.
    """

    def __init__(self, obsx, ref, azimuthal_anisotropic=False):
        self.obsx = obsx
        self.kmax = obsx.size
        self.ref = ref
        self.azimuthal_anisotropic = azimuthal_anisotropic

        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
            }

        self.wavetype, self.veltype = self.get_surftags(ref)

        if self.kmax > 60:
            raise Exception("Your observed data vector exceeds the maximum of 60 \
periods that is allowed in SurfDisp.")

        if self.azimuthal_anisotropic:
            if ref.startswith('l') or ref.endswith('gr'):
                raise Exception("should not happen")

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_surftags(self, ref):
        if ref == 'rdispgr':
            return (2, 1)

        elif ref == 'ldispgr':
            return (1, 1)

        elif ref == 'rdispph':
            return (2, 0)

        elif ref == 'ldispph':
            return (1, 0)
        else:
            tagerror = "Reference is not available in SurfDisp. If you defined \
a user Target, assign the correct reference (target.ref) or update the \
forward modeling plugin with target.update_plugin(MyForwardClass()).\n \
* Your ref was: %s\nAvailable refs are: rdispgr, ldispgr, rdispph, ldispph\n \
(r=rayleigh, l=love, gr=group, ph=phase)" % ref
            raise ReferenceError(tagerror)

    def get_modelvectors(self, h, vp, vs, rho):
        nlayer = len(h)
        thkm = np.zeros(200)
        thkm[:nlayer] = h

        vpm = np.zeros(200)
        vpm[:nlayer] = vp

        vsm = np.zeros(200)
        vsm[:nlayer] = vs

        rhom = np.zeros(200)
        rhom[:nlayer] = rho

        return thkm, vpm, vsm, rhom

    def simplify_model(self, h, vp, vs, rho, c1, c2):
        MAXLAYERS = 200 # according to the maximum number in pysurf96aa
        ns = int(np.ceil(len(h)/(MAXLAYERS-1)))+1
        hnew = np.zeros(MAXLAYERS)
        hnew[-1] = h[-1]
        vpnew = np.zeros(MAXLAYERS)
        vpnew[-1] = vp[-1]
        vsnew = np.zeros(MAXLAYERS)
        vsnew[-1] = vs[-1]
        rhonew = np.zeros(MAXLAYERS)
        rhonew[-1] = rho[-1]
        c1new = np.zeros(MAXLAYERS)
        c1new[-1] = c1[-1]
        c2new = np.zeros(MAXLAYERS)
        c2new[-1] = c2[-1]
        i = len(h)-1
        for inew in np.arange(MAXLAYERS-2,-1,-1):
            hnew[inew] = np.sum(h[i-ns:i])
            vpnew[inew] = np.average(vp[i-ns:i],weights=h[i-ns:i])
            vsnew[inew] = np.average(vs[i-ns:i],weights=h[i-ns:i])
            rhonew[inew] = np.average(rho[i-ns:i],weights=h[i-ns:i])
            c1new[inew] = np.average(c1[i-ns:i],weights=h[i-ns:i])
            c2new[inew] = np.average(c2[i-ns:i],weights=h[i-ns:i])
            i -= ns
            if i == inew:
                ns = 1
            else:
                ns = np.min([i-inew+1,int(np.ceil(i/(inew-1)))+1])
        if np.any(vsnew==0.) or np.any(np.isnan(vsnew)):
            raise Exception("should not happen!")
        return hnew, vpnew, vsnew, rhonew, c1new, c2new


    def run_model(self, h, vp, vs, rho, c1, c2, **params):
        """ The forward model will be run with the parameters below.

        thkm, vpm, vsm, rhom: model for dispersion calculation
        nlayer - I4: number of layers in the model
        iflsph - I4: 0 flat earth model, 1 spherical earth model
        iwave - I4: 1 Love wave, 2 Rayleigh wave
        mode - I4: ith mode of surface wave, 1 fundamental, 2 first higher, ...
        igr - I4: 0 phase velocity, > 0 group velocity
        kmax - I4: number of periods (t) for dispersion calculation
        t - period vector (t(NP))
        cg - output phase or group velocities (vector,cg(NP))

        """

        if len(h)>200:
            print("Warning: the model has more layers than the allowed maximum in SurfPy96AA (200). Model will be simplified.")
            h, vp, vs, rho, c1, c2 = self.simplify_model(h, vp, vs, rho, c1, c2)

        nlayer = len(h)
        h_vect, vp_vect, vs_vect, rho_vect = self.get_modelvectors(h, vp, vs, rho)

        iflsph = self.modelparams['flsph']
        mode = self.modelparams['mode']
        iwave = self.wavetype
        igr = self.veltype

        if self.kmax > 60:
            kmax = 60
            pers = self.obsx_int

        else:
            pers = np.zeros(60)
            kmax = self.kmax
            pers[:kmax] = self.obsx

        dispvel = np.zeros(60)  # result

        if self.azimuthal_anisotropic and (np.abs(c1)+np.abs(c2)!=0.).any():
            nrefine=1
            Lsen_Gsc,dcR_dA,dcR_dL,error = depthkernel(
                h_vect,vp_vect,vs_vect,rho_vect,nlayer,nrefine,iflsph,
                iwave,mode,igr,kmax,pers,dispvel)
            if error == 1:
                return np.nan, np.nan, np.nan, np.nan
            nlayers_refined = Lsen_Gsc.shape[1]
            # assumption: Vp anisotropy points in the same direction and has an amplitude 1.5 times stronger than Vs (Obreski et al. 2010, Bodin et al. 2016)
            # G = rho*vs*dvs, where dvs is the peak-to-peak azimuthal variation, i.e. dvs=2*anisoamp*vs (factor 2 because of peak-to-peak)
            # Gc = rho*vs*dvs*cos(2*psi2) = rho*vs*2*anisoamp*vs*cos(2*psi2)
            # input c1=anisoamp*cos(2*psi2) -> Gc = 2*rho*vs**2*c1, Gs = 2*rho*vs**2*c2
            Gc = 2*rho*vs**2*c1
            Gs = 2*rho*vs**2*c2
            # for the Bc,s terms, the additional 1.5 factor is needed
            Bc = 2*1.5*rho*vp**2*c1
            Bs = 2*1.5*rho*vp**2*c2
            # bring Gc, Gs, Bc, Bs to the same shape as dcR_dA and dcR_dL. Then integration according to eq. A3, A4 of Bodin et al. 2016
            C1 = np.sum(dcR_dA * np.reshape(np.tile(np.repeat(Bc[:-1],nrefine),kmax),(kmax,nlayers_refined)) +
                        dcR_dL * np.reshape(np.tile(np.repeat(Gc[:-1],nrefine),kmax),(kmax,nlayers_refined)),axis=1)
            C2 = np.sum(dcR_dA * np.reshape(np.tile(np.repeat(Bs[:-1],nrefine),kmax),(kmax,nlayers_refined)) +
                        dcR_dL * np.reshape(np.tile(np.repeat(Gs[:-1],nrefine),kmax),(kmax,nlayers_refined)),axis=1)
            # C1 and C2 are in absolute units, i.e. dCiso [km/s] deviations from the isotropic phase velocity
            # we need them in relative units as in C = Ciso * (1 + A2*cos(2*PSI2)), where A2=sqrt((C1/Ciso)**2 + (C2/Ciso)**2)
            return pers[:kmax], dispvel[:kmax], C1/dispvel[:kmax], C2/dispvel[:kmax]
        else:
            error = surfdisp96(h_vect, vp_vect, vs_vect, rho_vect, nlayer, 
                               iflsph, iwave, mode, igr, kmax, pers, dispvel)
            if error == 1:
                if self.azimuthal_anisotropic:
                    return np.nan, np.nan, np.nan, np.nan
                else:
                    return np.nan, np.nan
            if self.azimuthal_anisotropic:
                return pers[:kmax], dispvel[:kmax], np.zeros(kmax), np.zeros(kmax)
            else:
                return pers[:kmax], dispvel[:kmax]

