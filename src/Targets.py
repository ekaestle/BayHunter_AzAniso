# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

logger = logging.getLogger()


class ObservedData(object):
    """
    The observed data object only consists of x and y.

    x = continuous and monotone increasing vector
    y = y(x)

    """
    def __init__(self, x, y, yerr=None, c1=None, c1_err=None, c2=None, c2_err=None):
        self.x = x
        # in general, y is the same size as x. But in the case of azimuthal anisotropy
        # y is the phase velocity for different propagation directions and therefore
        # a longer vector (depending on SingleTarget.azimuths, see below)
        self.y = y
        self.yerr = yerr
        self.c1 = c1
        self.c1_err = c1_err
        self.c2 = c2
        self.c2_err = c2_err
        if c1 is not None and c2 is not None:
            self.aa_amp = np.sqrt(c1**2+c2**2)
            self.aa_ang = 0.5*np.arctan2(c2,c1)
        else:
            self.aa_amp = self.aa_ang = None

        if self.yerr is None or np.any(yerr<=0.) or np.any(np.isnan(yerr)):
            self.yerr = np.ones(x.size) * 1e-5
        if self.c1_err is None or np.any(c1_err<=0.) or np.any(np.isnan(c1_err)):
            self.c1_err = np.ones(x.size) * 1e-5
        if self.c2_err is None or np.any(c2_err<=0.) or np.any(np.isnan(c2_err)):
            self.c2_err = np.ones(x.size) * 1e-5

class ModeledData(object):
    """
    The modeled data object consists of x and y, which are initiated with nan,
    and will be computed during the inversion with the forward modeling tools.
    The plugins are python wrappers returning synthetic data, based on:
    RF: RFmini (Joachim Saul, GFZ, Posdam)
    SW: Surf96 (Rob Herrmann, St. Louis University, USA)

    You can easily update the plugin with your own code. Initiate the plugin
    with the necessary parameters and forward the instance to the
    update_plugin(instance) method. You can access this method through the
    SingleTarget object.

    The final method returning synthetic x and y data must be named
    self.run_model(h, vp, vs, rho, **kwargs). You can find a template with
    necessary plugin structure and method names in the defaults folder.
    Get inspired by the source code of the existing plugins.
    """
    def __init__(self, obsx, ref, azimuthal_anisotropic=False):
        rf_targets = ['prf', 'srf']
        swd_targets = ['rdispph', 'ldispph', 'rdispgr', 'ldispgr']
        self.azimuthal_anisotropic = azimuthal_anisotropic

        if ref in rf_targets:
            from BayHunter.rfmini_modrf import RFminiModRF
            self.plugin = RFminiModRF(obsx, ref)
            self.xlabel = 'Time in s'

        elif ref in swd_targets:
            from BayHunter.surf96_modsw import SurfDisp
            self.plugin = SurfDisp(obsx, ref, azimuthal_anisotropic=self.azimuthal_anisotropic)
            self.xlabel = 'Period in s'

        else:
            message = "Please provide a forward modeling plugin for your " + \
                "target.\nUse target.update_plugin(MyForwardClass())"
            logger.info(message)
            self.plugin = None
            self.xlabel = 'x'

        self.x = np.nan
        self.y = np.nan

    def update(self, plugin):
        self.plugin = plugin

    def calc_synth(self, h, vp, vs, **kwargs):
        """ Call forward modeling method of plugin."""
        if self.azimuthal_anisotropic:
            # y is the isotropic phase velocity
            # c1 is the x component of the anisotropic phase velocity
            # c2 is the y component ...
            self.x, self.y, self.c1, self.c2 = self.plugin.run_model(h, vp, vs, **kwargs)
            self.aa_amp = np.sqrt(self.c1**2+self.c2**2)
            self.aa_ang = 0.5*np.arctan2(self.c2,self.c1)
        else:
            self.x, self.y = self.plugin.run_model(h, vp, vs, **kwargs)

    #def get_anisotropy(self):
    #    if not self.azimuthal_anisotropic:
    #        return self.y,None,None
    #    def aniso_fitting_function(azi,A2,PHI2):
    #        return A2*np.cos(2*(azi-PHI2))
    #    def fit_anisotropy(azis,vels):
    #        popt,pcov = curve_fit(aniso_fitting_function,azis,vels,bounds=([0,-np.pi/2.],[20,np.pi/2.]))
    #        return popt
    #    phvels = np.column_stack(np.split(self.y,len(self.azimuths)))
    #    phvel_iso = np.zeros(len(self.x))
    #    aniso_amp = np.zeros(len(self.x))
    #    aniso_azi = np.zeros(len(self.x))
    #    for i,period in enumerate(self.x):
    #        phvel_iso[i] = np.mean(phvels[i])
    #        phvels_rel = (phvels[i]-np.mean(phvels[i]))/np.mean(phvels[i])*100.
    #        amp,azi = fit_anisotropy(self.azimuths,phvels_rel)
    #        aniso_amp[i] = amp
    #        aniso_azi[i] = azi
    #    return phvel_iso,aniso_amp,aniso_azi



# plt.figure()
# syny = np.split(t1.moddata.y,12)
# obsy = np.split(t1.obsdata.y,12)
# for ki in range(len(syny)):
#     plt.plot(syny[ki],'k')
#     plt.plot(obsy[ki])
# plt.savefig("testplot.pdf",bbox_inches='tight')

class Valuation(object):
    """
    Computation methods for likelihood and misfit are provided.
    The RMS misfit is only used for display in the terminal to get an estimate
    of the progress of the inversion.

    ONLY the likelihood is used for Bayesian inversion.
    """
    def __init__(self):
        self.corr_inv = None
        self.logcorr_det = None
        self.misfit = None
        self.likelihood = None

    @staticmethod
    def get_rms(yobs, ymod, angles=False):
        """Return root mean square."""
        ydiff = ymod - yobs
        if angles:
            ydiff = np.abs(ydiff)
            ydiff[ydiff>np.pi/2.] -= np.pi
        rms = np.sqrt(np.mean(ydiff**2))
        return rms

    @staticmethod
    def get_covariance_nocorr(sigma, size, yerr=None, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.

        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal.
        """
        c_inv = np.diag(np.ones(size)) / (sigma**2)
        logc_det = (2*size) * np.log(sigma)
        return c_inv, logc_det

    @staticmethod
    def get_covariance_nocorr_scalederr(sigma, size, yerr, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.

        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal. Errors are relatively scaled.
        """
        scaled_err = yerr / yerr.min()

        c_inv = np.diag(np.ones(size)) / (scaled_err * sigma**2)
        logc_det = (2*size) * np.log(sigma) + np.log(np.product(scaled_err)) 
        return c_inv, logc_det

    @staticmethod
    def get_corr_inv(corr, size):
        d = np.ones(size) + corr**2
        d[0] = d[-1] = 1
        e = np.ones(size-1) * -corr
        corr_inv = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
        return corr_inv

    def get_covariance_exp(self, corr, sigma, size, yerr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.

        The correlation between data points is represented by an EXPONENTIAL law.
        """
        c_inv = self.get_corr_inv(corr, size) / (sigma**2 * (1-corr**2))
        logc_det = (2*size) * np.log(sigma) + (size-1) * np.log(1-corr**2)

        return c_inv, logc_det

    def init_covariance_gauss(self, corr, size, rcond=None):
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                              (size, size))
        rmatrix = corr**(idx**2)

        if rcond is not None:
            self.corr_inv = np.linalg.pinv(rmatrix, rcond=rcond)
        else:
            self.corr_inv = np.linalg.inv(rmatrix)
        _, logdet = np.linalg.slogdet(rmatrix)
        self.logcorr_det = logdet

    def get_covariance_gauss(self, sigma, size, yerr=None, corr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.

        The correlation between data points is represented by a GAUSSIAN law.
        Consider this type of correlation if a gaussian filter was applied
        to compute RF. In this case, the inverse and log-determinant of the
        correlation matrix R is computed only once when initiating the chains.
        """
        c_inv = self.corr_inv / (sigma**2)
        logc_det = (2*size) * np.log(sigma) + self.logcorr_det
        return c_inv, logc_det

    @staticmethod
    def get_likelihood(yobs, ymod, c_inv, logc_det, angles=False):
        """Return log-likelihood."""
        ydiff = ymod - yobs
        if angles:
            ydiff = np.abs(ydiff)
            ydiff[ydiff>np.pi/2.] -= np.pi
        madist = (ydiff.T).dot(c_inv).dot(ydiff)  # Mahalanobis distance
        logL_part = -0.5 * (yobs.size * np.log(2*np.pi) + logc_det)
        logL = logL_part - madist / 2.

        return logL


class SingleTarget(object):
    """A SingleTarget object gathers observed and modeled data,
    and the valuation methods. It provides methods to calculate misfit and
    likelihood, and also a plotting method. These can be used when initiating
    and testing your targets.
    """
    def __init__(self, x, y, ref, yerr=None, c1=None, c1err=None, c2=None, c2err=None):
        self.ref = ref
        self.azimuthal_anisotropic=False
        if c1 is not None and c2 is not None:
            if ref.startswith('l'):
                print("Azimuthal anisotropy is currently not implemented for Love waves. The azimuthal anisotropy will be ignored for this target.")
                c1 = c2 = None
            elif ref.endswith('gr'):
                print("Azimuthal anisotropy is currently not implemented for group velocities. The azimuthal anisotropy will be ignored for this target.")
                c1 = c2 = None
        if c1 is not None and c2 is not None:
            #if np.max(np.abs(psi2azi))>2*np.pi:
            #    raise Exception("fast axis direction larger than 2pi; angles should be given in radians!")
            # normalization to [-pi/2,pi/2]
            #psi2azi_normed = 0.5*np.arctan2(np.sin(2*psi2azi),np.cos(2*psi2azi))
            self.azimuthal_anisotropic=True
        #else:
        #    psi2azi_normed=psi2azi
        self.obsdata = ObservedData(x=x, y=y, yerr=yerr, c1=c1, c1_err=c1err, c2=c2, c2_err=c2err)
        self.moddata = ModeledData(obsx=x, ref=ref, azimuthal_anisotropic=self.azimuthal_anisotropic)
        self.valuation = Valuation()

        logger.info("Initiated target: %s (ref: %s)"
                    % (self.__class__.__name__, self.ref))

    def update_plugin(self, plugin):
        self.moddata.update(plugin)

    def _moddata_valid(self):
        if not type(self.moddata.x) == np.ndarray:
            return False
        if not len(self.obsdata.x) == len(self.moddata.x):
            return False
        if not np.sum(self.obsdata.x - self.moddata.x) <= 1e-5:
            return False
        if not np.shape(self.obsdata.y) == np.shape(self.moddata.y):
            return False

        return True

    def calc_misfit(self):
        if not self._moddata_valid():
            self.valuation.misfit = 1e15
            return

        if self.azimuthal_anisotropic:
            self.valuation.misfit = 0.
            self.valuation.misfit += self.valuation.get_rms(
                self.obsdata.y, self.moddata.y)
            self.valuation.misfit += self.valuation.get_rms(
                self.obsdata.c1, self.moddata.c1)
            self.valuation.misfit += self.valuation.get_rms(
                self.obsdata.c2, self.moddata.c2)
        else:
            self.valuation.misfit = self.valuation.get_rms(
                self.obsdata.y, self.moddata.y)

    def calc_likelihood(self, noise):
        if not self._moddata_valid():
            self.valuation.likelihood = -1e15
            return

        corr,sigma1,sigma2,sigma3 = noise # 3 sigmas for the anisotropic stds
        if self.azimuthal_anisotropic:
            self.valuation.likelihood = 0.

            c_inv, logc_det = self.get_covariance(
                sigma=sigma1, size=self.obsdata.y.size, 
                yerr=self.obsdata.yerr, corr=corr)
            self.valuation.likelihood += self.valuation.get_likelihood(
                self.obsdata.y, self.moddata.y, c_inv, logc_det)

            c_inv, logc_det = self.get_covariance(
                sigma=sigma2, size=self.obsdata.c1.size, 
                yerr=self.obsdata.c1_err, corr=corr)
            self.valuation.likelihood += self.valuation.get_likelihood(
                self.obsdata.c1, self.moddata.c1, c_inv, logc_det)

            c_inv, logc_det = self.get_covariance(
                sigma=sigma3, size=self.obsdata.c2.size, 
                yerr=self.obsdata.c2_err, corr=corr)
            self.valuation.likelihood += self.valuation.get_likelihood(
                self.obsdata.c2, self.moddata.c2, c_inv, logc_det)

        else:
            c_inv, logc_det = self.get_covariance(
                sigma=sigma1, size=self.obsdata.y.size, 
                yerr=self.obsdata.yerr, corr=corr)
            self.valuation.likelihood = self.valuation.get_likelihood(
                self.obsdata.y, self.moddata.y, c_inv, logc_det)

    def plot(self, ax=None, mod=True, aziamp=True):
        axes=ax
        if ax is None:
            if self.azimuthal_anisotropic:
                fig, axes = plt.subplots(figsize=(20,3.5),ncols=3)
                ax = axes[0]
                ax2 = axes[1]
                ax3 = axes[2]
            else:
                fig,ax = plt.subplots()
                axes = ax
        elif type(ax)==type(np.array([])):
            ax = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]
        
        y = self.obsdata.y
        ax.errorbar(self.obsdata.x, y, yerr=self.obsdata.yerr,
                    label='obs', marker='x', ms=6, color='blue', lw=0.8,
                    elinewidth=0.7, zorder=1000)
        if self.azimuthal_anisotropic and aziamp:
            ax2.plot(self.obsdata.x,self.obsdata.aa_amp*100,
                     label='obs',marker='x',ms=6, color='blue', lw=0.8, zorder=1000)
            ax2.set_xlabel(self.moddata.xlabel)
            ax2.set_ylabel("Anisotropic amplitude in %")
            ax3.plot(self.obsdata.x,self.obsdata.aa_ang/np.pi*180,
                     marker='x',label='obs',ms=6, color='blue', zorder=1000)
            ax3.set_xlabel(self.moddata.xlabel)
            ax3.set_ylabel("Anisotropic azimuths (math. deg)")
            ax3.set_ylim(-90,90)
        elif self.azimuthal_anisotropic and not aziamp:
            ax2.errorbar(self.obsdata.x,self.obsdata.c1,yerr=self.obsdata.c1_err,
                         label='obs',marker='x',ms=6, color='blue', lw=0.8, zorder=1000)
            ax2.set_xlabel(self.moddata.xlabel)
            ax2.set_ylabel("C1")
            ax3.errorbar(self.obsdata.x,self.obsdata.c2,yerr=self.obsdata.c2_err,
                         label='obs',marker='x',ms=6, color='blue', lw=0.8, zorder=1000)
            ax3.set_xlabel(self.moddata.xlabel)
            ax3.set_ylabel("C2")

        if mod:
            ymod = self.moddata.y
            if self.azimuthal_anisotropic:
                ax2.plot(self.moddata.x, self.moddata.c1, label='mod',
                         marker='o',  ms=1, color='red', lw=0.7, alpha=0.5)
                ax3.plot(self.moddata.x, self.moddata.c2,'o', label='mod',
                         ms=1, color='red', alpha=0.5)
            ax.plot(self.moddata.x, ymod, label='mod',
                    marker='o',  ms=1, color='red', lw=0.7, alpha=0.5)

        ax.set_ylabel(self.ref)
        ax.set_xlabel(self.moddata.xlabel)

        return axes


class RayleighDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None, c1=None, c1err=None, c2=None, c2err=None):
        ref = 'rdispph'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr, c1=c1, c1err=c1err, c2=c2, c2err=c2err)


class RayleighDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None, c1=None, c1err=None, c2=None, c2err=None):
        ref = 'rdispgr'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr, c1=c1, c1err=c1err, c2=c2, c2err=c2err)


class LoveDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None, c1=None, c1err=None, c2=None, c2err=None):
        ref = 'ldispph'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr, c1=c1, c1err=c1err, c2=c2, c2err=c2err)


class LoveDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None, c1=None, c1err=None, c2=None, c2err=None):
        ref = 'ldispgr'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr, c1=c1, c1err=c1err, c2=c2, c2err=c2err)


class PReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, yerr=None):
        ref = 'prf'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class SReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, yerr=None):
        ref = 'srf'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class JointTarget(object):
    """A JointTarget object contains a list of SingleTargets and is responsible
    for computing the joint likelihood, given all model parameters."""
    def __init__(self, targets):
        self.targets = targets  # list of SingleTargets
        self.ntargets = len(targets)

    def get_misfits(self):
        """Compute misfit by summing target misfits.
        Keep targets' individual misfits for comparison purposes."""
        misfits = [target.valuation.misfit for target in self.targets]
        jointmisfit = np.sum(misfits)
        return np.concatenate((misfits, [jointmisfit]))

    def evaluate(self, h, vp, vs, noise, **kwargs):
        """This evaluation method basically evaluates the given model.
        It computes the jointmisfit, and more important the jointlikelihoods.
        The jointlikelihood (here called the proposallikelihood) is the sum
        of the log-likelihoods from each target."""
        rho = kwargs.pop('rho', vp * 0.32 + 0.77)
        psi2amp = kwargs.pop('psi2amp')
        psi2azi = kwargs.pop('psi2azi')
        c1 = psi2amp * np.cos(2*psi2azi)
        c2 = psi2amp * np.sin(2*psi2azi)

        logL = 0
        for n, target in enumerate(self.targets):
            target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho,
                                      c1=c1, c2=c2, **kwargs)

            if not target._moddata_valid():
                self.proposallikelihood = -1e15
                self.proposalmisfits = [1e15]*(self.ntargets+1)
                return

            target.calc_misfit()

            target_noise = noise[4*n:4*n+4]
            target.calc_likelihood(target_noise)
            logL += target.valuation.likelihood

        self.proposallikelihood = logL
        self.proposalmisfits = self.get_misfits()

    def plot_obsdata(self, ax=None, mod=False, aziamp=True):
        """Return subplot of all targets."""
        if len(self.targets) == 1:
            if ax is None:
                ax = self.targets[0].plot(ax=ax, mod=mod, aziamp=aziamp)
            if type(ax)==type(np.array([])):
                fig = ax[0].figure
                ax[0].legend()
            else:
                fig = ax.figure
                ax.legend()

        else:
            if ax is None:
                fig, ax = plt.subplots(nrows=self.ntargets,ncols=3,
                                       figsize=(20, 3.2*self.ntargets))
            else:
                fig = ax[0].figure

            for i, target in enumerate(self.targets):
                ax[i] = target.plot(ax=ax[i], mod=mod, aziamp=aziamp)

            if len(ax.shape)>1:
                han, lab = ax[0][0].get_legend_handles_labels()
                ax[0][0].legend(han, lab)
            else:
                han, lab = ax[0].get_legend_handles_labels()
                ax[0].legend(han, lab)

        return fig, ax
