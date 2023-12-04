# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import copy
import time
import numpy as np
import os.path as op
from scipy.interpolate import interp1d

from BayHunter import Model, ModelMatrix
from BayHunter import utils

import logging
logger = logging.getLogger()


PAR_MAP = {'vsmod': 0, 'zvmod': 1, 'birth': 2, 'death': 2,
           'noise': 3, 'vpvs': 4, 'aniso_birth': 5, 'aniso_death': 5,
           'aniso_ampmod': 6, 'aniso_dirmod': 7}


class SingleChain(object):

    def __init__(self, targets, chainidx=0, initparams={}, modelpriors={},
                 sharedmodels=None, sharedmisfits=None, sharedlikes=None,
                 sharednoise=None, sharedvpvs=None, sharedtemperatures=None,
                 sharedlikes_current=None,random_seed=None, nmodels=None):

        self.chainidx = chainidx
        self.rstate = np.random.RandomState(random_seed)
        self.nmodels = nmodels

        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.initparams.update(initparams)
        self.priors.update(modelpriors)
        self.dv = (self.priors['vs'][1] - self.priors['vs'][0])

        self.nchains = self.initparams['nchains']
        self.station = self.initparams['station']

        # experimental fixing the velocity model and only fitting the anisotropy
        self.fixedvelmodel = False

        # set targets and inversion specific parameters
        self.targets = targets

        # set parameters
        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iterations = self.iter_phase1 + self.iter_phase2
        self.iiter = -self.iter_phase1
        self.lastmoditer = self.iiter

        self.propdist = np.array(self.initparams['propdist'])
        self.acceptance = self.initparams['acceptance']
        self.thickmin = self.initparams['thickmin']
        self.maxlayers = int(self.priors['layers'][1]) + 1

        self.lowvelperc = self.initparams['lvz']
        self.highvelperc = self.initparams['hvz']
        self.mantle = self.priors['mantle']

        # chain models
        self._init_chainarrays(sharedmodels, sharedmisfits, sharedlikes,
                               sharednoise, sharedvpvs, sharedtemperatures,
                               sharedlikes_current)

        # init model and values
        self._init_model_and_currentvalues()

        # set the modelupdates
        self.modelmods = ['vsmod', 'zvmod', 'birth', 'death']
        self.noisemods = [] if len(self.noiseinds) == 0 else ['noise']
        self.vpvsmods = [] if type(self.priors['vpvs']) == float else ['vpvs']
        self.anisomods = [] if self.initparams['azimuthal_anisotropy'] == False else ['aniso_birth','aniso_death','aniso_ampmod','aniso_dirmod']
        self.modifications = self.modelmods + self.noisemods + self.vpvsmods + self.anisomods

        nmods = len(self.propdist)
        if self.initparams['azimuthal_anisotropy']:
            # +3 because of anisotropic steps (birth/death, anisamp, anisoazi)
            nmods += 3
        self.accepted = np.zeros(nmods)
        self.proposed = np.zeros(nmods)
        self.acceptancerate = np.ones((nmods,100))

        # start at minus burnin iterations
        self.iiter = -self.iter_phase1


# init model and misfit / likelihood

    def _init_model_and_currentvalues(self):
        ivpvs = self.draw_initvpvs()
        self.currentvpvs = ivpvs
        imodel = self.draw_initmodel()
        # self.currentmodel = imodel
        inoise, corrfix = self.draw_initnoiseparams()
        # self.currentnoise = inoise

        rcond = self.initparams['rcond']
        self.set_target_covariance(corrfix[::4], inoise[::4], rcond)

        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(imodel, ivpvs, self.mantle)
        self.targets.evaluate(h=h, vp=vp, vs=vs, noise=inoise,
                              psi2amp=psi2amp,psi2azi=psi2azi)

        # self.currentmisfits = self.targets.proposalmisfits
        # self.currentlikelihood = self.targets.proposallikelihood

        logger.debug((vs, h))

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(imodel, inoise, ivpvs)
        self.append_currentmodel()

    def draw_initmodel(self):
        keys = self.priors.keys()
        zmin, zmax = self.priors['z']
        vsmin, vsmax = self.priors['vs']
        layers = np.min(self.priors['layers']) + 1  # half space

        for i in range(10000):
            vs = self.rstate.uniform(low=vsmin, high=vsmax, size=layers)
            vs.sort()

            if (self.priors['mohoest'] is not None and layers > 1):
                mean, std = self.priors['mohoest']
                moho = self.rstate.normal(loc=mean, scale=np.min([10,std]))
                tmp_z = self.rstate.uniform(1, np.min([5, moho]))  # 1-5
                tmp_z_vnoi = np.array([moho-tmp_z, moho+tmp_z])

                if (layers - 2) == 0:
                    z_vnoi = tmp_z_vnoi
                else:
                    if self.priors['triangular_zprop']:
                        z_vnoi = self.rstate.triangular(zmin, zmin, zmax, size=(layers - 2))
                    else:
                        z_vnoi = self.rstate.uniform(low=zmin, high=zmax, size=(layers - 2))     
                    z_vnoi = np.concatenate((tmp_z_vnoi,z_vnoi))

            else:  # no moho estimate
                if self.priors['triangular_zprop']:
                    z_vnoi = self.rstate.triangular(zmin, zmin, zmax, size=layers)
                else:
                    z_vnoi = self.rstate.uniform(low=zmin, high=zmax, size=layers)

            z_vnoi.sort()
            vs[(z_vnoi>50.)*(vs<4.0)] = 4.0
            model = np.concatenate((vs, z_vnoi, np.zeros(len(vs)), np.zeros(len(vs))))

            if self._validmodel(model):
                return model
            else:
                continue

        raise Exception("no valid starting model found!")

    def draw_initnoiseparams(self):
        # for each target the noiseparams are (corr and sigma)
        noiserefs = ['noise_corr', 'noise_sigma', 'noise_sigma_c1', 'noise_sigma_c2'] 
            # 3 times noise sigma for the isotropic and the 2 anisotropic parameters
        init_noise = np.ones(len(self.targets.targets)*4) * np.nan
        corrfix = np.zeros(len(self.targets.targets)*4, dtype=bool)

        self.noisepriors = []
        for i, target in enumerate(self.targets.targets):
            for j, noiseref in enumerate(noiserefs):
                idx = (4*i)+j
                noiseprior = self.priors[target.noiseref + noiseref]

                if j>1 and not target.azimuthal_anisotropic:
                    corrfix[idx] = True
                    init_noise[idx] = np.nan
                elif type(noiseprior) in [int, float, np.float64]:
                    corrfix[idx] = True
                    init_noise[idx] = noiseprior
                else:
                    init_noise[idx] = self.rstate.uniform(
                        low=noiseprior[0], high=noiseprior[1])

                self.noisepriors.append(noiseprior)
 
        self.noiseinds = np.where(corrfix == 0)[0]
        if len(self.noiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')

        return init_noise, corrfix

    def draw_initvpvs(self):
        if type(self.priors['vpvs']) == float:
            return self.priors['vpvs']

        vpvsmin, vpvsmax = self.priors['vpvs']
        return self.rstate.uniform(low=vpvsmin, high=vpvsmax)

    def set_target_covariance(self, corrfix, noise_corr, rcond=None):
        # SWD noise hyper-parameters: if corr is not 0, the correlation of data
        # points assumed will be exponential.
        # RF noise hyper-parameters: if corr is not 0, but fixed, the
        # correlation between data points will be assumed gaussian (realistic).
        # if the prior for RFcorr is a range, the computation switches
        # to exponential correlated noise for RF, as gaussian noise computation
        # is too time expensive because of computation of inverse and
        # determinant each time _corr is perturbed

        for i, target in enumerate(self.targets.targets):
            target_corrfix = corrfix[i]
            target_noise_corr = noise_corr[i]

            if not target_corrfix:
                # exponential for each target
                target.get_covariance = target.valuation.get_covariance_exp
                continue

            if (target_noise_corr == 0 and np.any(np.isnan(target.obsdata.yerr))):
                # diagonal for each target, corr inrelevant for likelihood, rel error
                target.get_covariance = target.valuation.get_covariance_nocorr
                continue

            elif target_noise_corr == 0:
                # diagonal for each target, corr inrelevant for likelihood
                target.get_covariance = target.valuation.get_covariance_nocorr_scalederr
                continue

            # gauss for RF
            if target.noiseref == 'rf':
                size = target.obsdata.x.size
                target.valuation.init_covariance_gauss(
                    target_noise_corr, size, rcond=rcond)
                target.get_covariance = target.valuation.get_covariance_gauss

            # exp for noise_corr
            elif target.noiseref == 'swd':
                target.get_covariance = target.valuation.get_covariance_exp

            else:
                message = 'The noise correlation automatically defaults to the \
exponential law. Explicitly state a noise reference for your user target \
(target.noiseref) if wished differently.'
                logger.info(message)
                target.noiseref == 'swd'
                target.get_covariance = target.valuation.get_covariance_exp

    def _init_chainarrays(self, sharedmodels, sharedmisfits, sharedlikes,
                          sharednoise, sharedvpvs, sharedtemperatures,
                          sharedlikes_current):
        """from shared arrays"""
        ntargets = self.targets.ntargets
        chainidx = self.chainidx
        nchains = self.nchains

        msize = self.nmodels * self.maxlayers * 4
        nsize = self.nmodels * ntargets * 4
        missize = self.nmodels * (ntargets + 1)
        dtype = np.float32

        models = np.frombuffer(sharedmodels, dtype=dtype).\
            reshape((nchains, msize))
        misfits = np.frombuffer(sharedmisfits, dtype=dtype).\
            reshape((nchains, missize))
        likes = np.frombuffer(sharedlikes, dtype=dtype).\
            reshape((nchains, self.nmodels))
        noise = np.frombuffer(sharednoise, dtype=dtype).\
            reshape((nchains, nsize))
        vpvs = np.frombuffer(sharedvpvs, dtype=dtype).\
            reshape((nchains, self.nmodels))
        temperatures = np.frombuffer(sharedtemperatures, dtype=dtype).\
            reshape((nchains,self.iterations))
        
        self.currentlike_shared = np.frombuffer(
            sharedlikes_current, dtype=dtype)
        self.chainmodels = models[chainidx].reshape(
            self.nmodels, self.maxlayers*4)
        self.chainmisfits = misfits[chainidx].reshape(
            self.nmodels, ntargets+1)
        self.chainlikes = likes[chainidx]
        self.chainnoise = noise[chainidx].reshape(
            self.nmodels, ntargets*4)
        self.chainvpvs = vpvs[chainidx]
        self.temperatures = temperatures[chainidx]
        self.chainiter = np.ones(self.chainlikes.size) * np.nan

        self.temperature = self.temperatures[0]


# update current model (change layer number and values)

    def _model_layerbirth(self, model):
        """
        Draw a random voronoi nucleus depth from z and assign a new Vs.

        The new Vs is based on the before Vs value at the drawn z_vnoi
        position (self.propdist[2]).
        """
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)

        # new voronoi depth
        zmin, zmax = self.priors['z']
        if self.priors['triangular_zprop']:
            z_birth = self.rstate.triangular(zmin,zmin,zmax)
        else:
            z_birth = self.rstate.uniform(low=zmin, high=zmax)

        ind = np.argmin((abs(z_vnoi - z_birth)))  # closest z_vnoi
        vs_before = vs_vnoi[ind]
        vs_birth = vs_before + self.rstate.normal(0, self.propdist[2])

        z_new = np.concatenate((z_vnoi, [z_birth]))
        vs_new = np.concatenate((vs_vnoi, [vs_birth]))
        #if self.fixedvelmodel:
        #    psi2amp_new = np.concatenate((psi2amp,[self.rstate.uniform(0,  0.1)]))
        #else:
        psi2amp_new = np.concatenate((psi2amp,[0.]))
        psi2azi_new = np.concatenate((psi2azi,[self.rstate.uniform(0,np.pi)]))
        
        self.dvs2 = np.square(vs_birth - vs_before)
        return np.concatenate((vs_new, z_new, psi2amp_new, psi2azi_new))

    def _model_layerdeath(self, model):
        """
        Remove a random voronoi nucleus depth from model. Delete corresponding
        Vs from model.
        """
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        #if self.fixedvelmodel:
        #    death_candidates = np.arange(len(psi2amp))
        #else:
        death_candidates = np.where(psi2amp==0.)[0] # only death of isotropic cells
        if len(death_candidates)==0:
            return np.zeros(len(model))
        ind_death = self.rstate.choice(death_candidates)
        z_before = z_vnoi[ind_death]
        vs_before = vs_vnoi[ind_death]

        z_new = np.delete(z_vnoi, ind_death)
        vs_new = np.delete(vs_vnoi, ind_death)
        psi2amp_new = np.delete(psi2amp, ind_death)
        psi2azi_new = np.delete(psi2azi, ind_death)

        ind = np.argmin((abs(z_new - z_before)))
        vs_after = vs_new[ind]
        self.dvs2 = np.square(vs_after - vs_before)
        return np.concatenate((vs_new, z_new, psi2amp_new, psi2azi_new))

    def _model_vschange(self, model):
        """Randomly choose a layer to change Vs with Gauss distribution."""
        ind = self.rstate.randint(0, model.size / 4)
        vs_mod = self.rstate.normal(0, self.propdist[0])
        model[ind] = model[ind] + vs_mod
        return model

    def _model_zvnoi_move(self, model):
        """Randomly choose a layer to change z_vnoi with Gauss distribution."""
        ind = self.rstate.randint(model.size / 4, model.size / 2)
        z_mod = self.rstate.normal(0, self.propdist[1])
        model[ind] = model[ind] + z_mod
        return model
    
    def _model_azianiso_ampchange(self, model):
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        ind = np.where(psi2amp>0.)[0] # only update layers with anisotropy
        if len(ind)==0:
            return np.zeros(len(model))
        ind = self.rstate.choice(ind)
        ampmod = self.rstate.normal(0, 0.01)
        psi2amp[ind] += ampmod
        #if self.fixedvelmodel and psi2amp[ind]<0.:
        #    psi2amp[ind] = 1e-5
        return np.concatenate((vs_vnoi, z_vnoi, psi2amp, psi2azi))
        
    def _model_azianiso_azichange(self, model):
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        ind = np.where(psi2amp>0.)[0]
        if len(ind)==0:
            return np.zeros(len(model))
        ind = self.rstate.choice(ind)
        azimod = self.rstate.normal(0, np.pi/10.)
        psi2azi[ind] += azimod
        if psi2azi[ind]<0:
            psi2azi[ind] += np.pi
        elif psi2azi[ind] >= np.pi:
            psi2azi[ind] -= np.pi
        return np.concatenate((vs_vnoi, z_vnoi, psi2amp, psi2azi))
    
    def _model_azianiso_birth(self, model):
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        ind = np.where(psi2amp==0.)[0]
        if len(ind)==0:
            return np.zeros(len(model))
        ind = self.rstate.choice(ind)
        psi2amp[ind] = self.rstate.uniform(0,0.1)
        psi2azi[ind] = self.rstate.uniform(0,np.pi)
        return np.concatenate((vs_vnoi, z_vnoi, psi2amp, psi2azi))

    def _model_azianiso_death(self, model):
        n, vs_vnoi, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        ind = np.where(psi2amp>0.)[0]
        if len(ind)==0:
            return np.zeros(len(model))
        ind = self.rstate.choice(ind)
        psi2amp[ind] = 0.
        psi2azi[ind] = self.rstate.uniform(0,np.pi)
        return np.concatenate((vs_vnoi, z_vnoi, psi2amp, psi2azi))        


    def _get_modelproposal(self, modify):
        model = copy.copy(self.currentmodel)

        if modify == 'vsmod':
            propmodel = self._model_vschange(model)
        elif modify == 'zvmod':
            propmodel = self._model_zvnoi_move(model)
        elif modify == 'birth':
            propmodel = self._model_layerbirth(model)
        elif modify == 'death':
            propmodel = self._model_layerdeath(model)
        elif modify == 'aniso_birth':
            propmodel = self._model_azianiso_birth(model)
        elif modify == 'aniso_death':
            propmodel = self._model_azianiso_death(model)
        elif modify == 'aniso_ampmod':
            propmodel = self._model_azianiso_ampchange(model)
        elif modify == 'aniso_dirmod':
            propmodel = self._model_azianiso_azichange(model)

        return self._sort_modelproposal(propmodel)

    def _sort_modelproposal(self, model):
        """
        Return the sorted proposal model.

        This method is necessary, if the z_vnoi from the new proposal model
        are not ordered, i.e. if one z_vnoi value is added or strongly modified.
        """
        n, vs, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        if np.all(np.diff(z_vnoi) > 0):   # monotone increasing
            return model
        else:
            ind = np.argsort(z_vnoi)
            model_sort = np.concatenate((vs[ind], z_vnoi[ind], psi2amp[ind], psi2azi[ind]))
        return model_sort

    def _validmodel(self, model):
        """
        Check model before the forward modeling.

        - The model must contain all values > 0.
        - The layer thicknesses must be at least thickmin km.
        - if lvz: low velocity zones are allowed with the deeper layer velocity
           no smaller than (1-perc) * velocity of layer above.
        - ... and some other constraints. E.g. vs boundaries (prior) given.
        """

        _,_, z_vnoi, psi2amp, psi2azi = Model.split_modelparams(model)
        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(model, self.currentvpvs, self.mantle)

        # check that psi2amp is within the prior range, 0-10% anisotropy (hardcoded currently)
        if np.any(psi2amp>0.1) or np.any(psi2amp<0.):
            return False

        # check whether nlayers lies within the prior
        layermin = self.priors['layers'][0]
        layermax = self.priors['layers'][1]
        layermodel = (h.size - 1)
        if not (layermodel >= layermin and layermodel <= layermax):
            logger.debug("chain%d: model- nlayers not in prior"
                         % self.chainidx)
            return False

        # check whether interfaces lie within prior
        zmin = self.priors['z'][0]
        zmax = self.priors['z'][1]
        z = np.cumsum(h)
        if np.any(z < zmin) or np.any(z > zmax):
            logger.debug("chain%d: model- z not in prior"
                         % self.chainidx)
            return False

        if self.fixedvelmodel: # other checks not necessary
            return True

        # make sure there is a Moho step in the model
        if self.priors['mohoest'] is not None:
            crustidx = np.where(vs<4.1)[0]
            mantleidx = np.where(vs>=4.1)[0]
            mohoest, std = self.priors['mohoest']
            # at least 1 crustal layer, no velocities above 4.2 in the crust
            if len(crustidx)<1 or (np.diff(crustidx)>1).any():
                return False
            # make sure that Moho depth is within the prior limits
            z_moho = np.sum(h[crustidx])
            if z_moho>mohoest+std or z_moho<mohoest-std:
                return False
            # in the mantle, all velocities should be larger 4.1, beneath the Moho > 4.2
            if len(mantleidx)>0:
                if vs[mantleidx[0]]<4.2 or (vs[mantleidx[0]:]<4.1).any():
                    return False
                if (np.abs(np.diff(vs[mantleidx]))>0.3).any():
                    return False

        # check model for layers with thicknesses of smaller thickmin
        if self.initparams['relative_thickmin']:
            thickmin = self.thickmin * z_vnoi[:-1]
        else:
            thickmin = self.thickmin
        if np.any(h[:-1] < thickmin):
            logger.debug("chain%d: thicknesses are not larger than thickmin"
                         % self.chainidx)
            return False

        # check whether vs lies within the prior
        vsmin = self.priors['vs'][0]
        vsmax = self.priors['vs'][1]
        if np.any(vs < vsmin) or np.any(vs > vsmax):
            logger.debug("chain%d: model- vs not in prior"
                         % self.chainidx)
            return False

        # instead of percent, use absolute velocities.
        if self.lowvelperc is not None:
            # check model for low velocity zones. If larger than perc, then
            # compvels must be positive
            if np.any(np.diff(vs) < -np.abs(self.lowvelperc)):
                return False

        if self.highvelperc is not None:
            # check model for high velocity zones. If larger than perc, then
            # compvels must be positive.
            if np.any(np.diff(vs) > np.abs(self.highvelperc)):
                return False

        if self.iiter < -int(self.iter_phase1/3):
            # allow no low velocity zones for the first 1/3 of burnin iterations
            if np.any(np.diff(vs) < 0.):
                return False

        return True

    def _get_hyperparameter_proposal(self):
        noise = copy.copy(self.currentnoise)
        ind = self.rstate.choice(self.noiseinds)

        noise_mod = self.rstate.normal(0, self.propdist[3])
        noise[ind] = noise[ind] + noise_mod
        return noise

    def _validnoise(self, noise):
        for idx in self.noiseinds:
            if noise[idx] < self.noisepriors[idx][0] or \
                    noise[idx] > self.noisepriors[idx][1]:
                return False
        return True

    def _get_vpvs_proposal(self):
        vpvs = copy.copy(self.currentvpvs)
        vpvs_mod = self.rstate.normal(0, self.propdist[4])
        vpvs = vpvs + vpvs_mod
        return vpvs

    def _validvpvs(self, vpvs):
        # only works if vpvs-priors is a range
        if vpvs < self.priors['vpvs'][0] or \
                vpvs > self.priors['vpvs'][1]:
            return False
        return True


# accept / save current models

    def adjust_propdist(self):
        """
        Modify self.propdist to adjust acceptance rate of models to given
        percentace span: increase or decrease by five percent.
        """
        #with np.errstate(invalid='ignore'):
        #    acceptrate = self.accepted / self.proposed * 100
        acceptrate = np.sum(self.acceptancerate,axis=1)

        # minimum distribution width forced to be not less than 1 m/s, 1 m
        # actually only touched by vs distribution
        propdistmin = np.full(acceptrate.size, 0.01)

        for i, rate in enumerate(acceptrate):
            if i >= len(self.initparams['propfixed']):
                break
            if np.isnan(rate) or self.initparams['propfixed'][i]:
                # only if not inverted for
                continue
            if rate < self.acceptance[0]:
                new = self.propdist[i] * 0.95
                if new < propdistmin[i]:
                    new = propdistmin[i]
                self.propdist[i] = new

            elif rate > self.acceptance[1]:
                self.propdist[i] = self.propdist[i] * 1.05
            else:
                pass

    def get_acceptance_probability(self, modify):
        """
        Acceptance probability will be computed dependent on the modification.

        Parameterization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.

        Model dimension alteration (layer birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """
        if modify in ['vsmod', 'zvmod', 'noise', 'vpvs'] or 'aniso' in modify or self.fixedvelmodel:
            # only velocity or thickness changes are made
            # also used for noise changes
            alpha = self.targets.proposallikelihood - self.currentlikelihood
            alpha *= 1./self.temperature

        elif modify in ['birth', ]:
            theta = self.propdist[2]  # Gaussian distribution
            # self.dvs2 = delta vs square = np.square(v'_(k+1) - v_(i))
            A = (theta * np.sqrt(2 * np.pi)) / self.dv
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood
            if len(self.anisomods)>0: # see E10 in Bodin et al. 2016 (imaging anisotropic...)
                D = (len(self.currentmodel)/4) / (len(self.currentmodel)/4 + 1)
            else:
                D = 0

            alpha = np.log(A) + B + C * 1./self.temperature + D

        elif modify in ['death', ]:
            theta = self.propdist[2]  # Gaussian distribution
            # self.dvs2 = delta vs square = np.square(v'_(j) - v_(i))
            A = self.dv / (theta * np.sqrt(2 * np.pi))
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood
            if len(self.anisomods)>0: # see E10 in Bodin et al. 2016 (imaging anisotropic...)
                D = (len(self.currentmodel)/4 + 1) / (len(self.currentmodel)/4)
            else:
                D = 0
                
            alpha = np.log(A) - B + C * 1./self.temperature + D

        return alpha

    def accept_as_currentmodel(self, model, noise, vpvs):
        """Assign currentmodel and currentvalues to self."""
        self.currentmisfits = self.targets.proposalmisfits
        self.currentlikelihood = self.targets.proposallikelihood
        self.currentlike_shared[self.chainidx] = self.currentlikelihood
        self.currentmodel = model
        self.currentnoise = noise
        self.currentvpvs = vpvs

    def append_currentmodel(self):
        """Append currentmodel to chainmodels and values."""
        self.chainmodels[self.n, :self.currentmodel.size] = self.currentmodel
        self.chainmisfits[self.n, :] = self.currentmisfits
        self.chainlikes[self.n] = self.currentlikelihood
        self.chainnoise[self.n, :] = self.currentnoise
        self.chainvpvs[self.n] = self.currentvpvs

        self.chainiter[self.n] = self.iiter
        self.lastmoditer = self.iiter
        self.n += 1

# run optimization

    def iterate(self):

        # set starttime
        if self.iiter == -self.iter_phase1:
            self.tstart = time.time()
            self.tnull = time.time()

        # get current temperature
        #if self.iiter < -self.iter_phase1*0.5:
        #    self.temperature = 1.
        #else:
        self.temperature = self.temperatures[self.iiter+self.iter_phase1]

        #if True: # extratest
        #    if self.fixedvelmodel:
        #        proposalmodel, vp, vs, h, psi2amp, psi2azi = self._get_fixedmodel_params(self.currentmodel, self.currentvpvs, self.mantle)
        #    else:
        #        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(self.currentmodel, self.currentvpvs, self.mantle)
        #    self.targets.evaluate(h=h, vp=vp, vs=vs, noise=self.currentnoise,
        #                          psi2amp=psi2amp,psi2azi=psi2azi)
        #    if self.targets.proposallikelihood != self.currentlikelihood:
        #        raise Exception(self.iiter,"very bad!")
            

        if self.iiter < (-self.iter_phase1 + (self.iterations * 0.01)) and not self.fixedvelmodel:
            # only allow vs and z modifications the first 1 % of iterations
            modify = self.rstate.choice(['vsmod', 'zvmod'] + self.noisemods +
                                        self.vpvsmods)
        else:
            modify = self.rstate.choice(self.modifications)
            while 'aniso' in modify and self.iiter<-int(self.iter_phase1/2) and not self.fixedvelmodel:
                modify = self.rstate.choice(self.modifications)

        if modify in self.modelmods or modify in self.anisomods:
            proposalmodel = self._get_modelproposal(modify)
            proposalnoise = self.currentnoise
            proposalvpvs = self.currentvpvs
            if not self._validmodel(proposalmodel):
                proposalmodel = None

        elif modify in self.noisemods:
            proposalmodel = self.currentmodel
            proposalnoise = self._get_hyperparameter_proposal()
            proposalvpvs = self.currentvpvs
            if not self._validnoise(proposalnoise):
                proposalmodel = None

        elif modify == 'vpvs':
            proposalmodel = self.currentmodel
            proposalnoise = self.currentnoise
            proposalvpvs = self._get_vpvs_proposal()
            if not self._validvpvs(proposalvpvs):
                proposalmodel = None

        if proposalmodel is None:
            # If not a valid proposal model and noise params are found,
            # leave self.iterate and try with another modification
            # should not occur often.
            logger.debug('Not able to find a proposal for %s' % modify)

        else:
            # compute synthetic data and likelihood, misfit
            if self.fixedvelmodel:
                proposalmodel, vp, vs, h, psi2amp, psi2azi = self._get_fixedmodel_params(proposalmodel, proposalvpvs, self.mantle)
            else:
                vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(proposalmodel, proposalvpvs, self.mantle)
            self.targets.evaluate(h=h, vp=vp, vs=vs, noise=proposalnoise,
                                  psi2amp=psi2amp,psi2azi=psi2azi)

            paridx = PAR_MAP[modify]
            self.proposed[paridx] += 1

            # Replace self.currentmodel with proposalmodel with acceptance
            # probability alpha. Accept candidate sample (proposalmodel)
            # with probability alpha, or reject it with probability (1 - alpha).
            # these are log values ! alpha is log.
            u = np.log(self.rstate.uniform(0, 1))
            alpha = self.get_acceptance_probability(modify)

            # #### _____________________________________________________________
            if u < alpha:
                # always the case if self.jointlike > self.bestlike (alpha>1)
                self.accept_as_currentmodel(proposalmodel, proposalnoise, proposalvpvs)
                # avoid that the preallocated array is overfilled if the acceptancerate is too high
                if np.sum(self.acceptancerate[paridx])>self.acceptance[1]:
                    if np.random.uniform() < self.acceptance[1]/np.sum(self.acceptancerate[paridx]):
                        self.append_currentmodel()
                else:
                    self.append_currentmodel()
                self.accepted[paridx] += 1
                self.acceptancerate[paridx][0] = 1
                self.acceptancerate[paridx] = np.roll(self.acceptancerate[paridx],1)
            else:
                self.acceptancerate[paridx][0] = 0
                self.acceptancerate[paridx] = np.roll(self.acceptancerate[paridx],1)

        # print inversion status information
        if self.iiter % 50000 == 0 or self.iiter == -self.iter_phase1:
            runtime = time.time() - self.tnull
            current_iterations = self.iiter + self.iter_phase1

            if current_iterations > 0:
                acceptrate = np.sum(self.acceptancerate,axis=1)
                rates = ''
                for rate in acceptrate:
                    rates += '%2d ' %rate
                acceptrate_total = float(self.n) / current_iterations * 100.

                logger.info('Chain %3d (T=%5.2f): %6d %5d + hs %8.3f\t%9d |%6.1f s  | %s (%.1f%%)' % (
                    self.chainidx,self.temperature,
                    self.lastmoditer, self.currentmodel.size/4 - 1,
                    self.currentmisfits[-1], self.currentlikelihood,
                    runtime, rates, acceptrate_total))

            self.tnull = time.time()

        # stabilize model acceptance rate
        if self.iiter % 1000 == 0:
            if np.any(self.proposed > 100):
                self.adjust_propdist()
                

        # when the burnin phase is finished, start the collection
        # with the last accepted model
        if self.iiter == 0:
            if self.lastmoditer != 0:
                self.lastmoditer = self.iiter
                self.append_currentmodel()

        self.iiter += 1

        # set endtime
        if self.iiter == self.iter_phase2:
            self.tend = time.time()


    """
    def run_chain(self): #unused, since we want to run all chains in parallel
        t0 = time.time()
        self.tnull = time.time()
        self.iiter = -self.iter_phase1
    
        self.modelmods = ['vsmod', 'zvmod', 'birth', 'death']
        self.noisemods = [] if len(self.noiseinds) == 0 else ['noise']
        self.vpvsmods = [] if type(self.priors['vpvs']) == float else ['vpvs']
        self.modifications = self.modelmods + self.noisemods + self.vpvsmods
    
        self.accepted = np.zeros(len(self.propdist))
        self.proposed = np.zeros(len(self.propdist))
    
        while self.iiter < self.iter_phase2:
            self.iterate()
    """

    def finalize(self):

        runtime = (self.tend - self.tstart)

        # update chain values (eliminate nan rows)
        self.chainmodels = self.chainmodels[:self.n, :]
        self.chainmisfits = self.chainmisfits[:self.n, :]
        self.chainlikes = self.chainlikes[:self.n]
        self.chainnoise = self.chainnoise[:self.n, :]
        self.chainvpvs = self.chainvpvs[:self.n]
        self.chainiter = self.chainiter[:self.n]

        # only consider models after burnin phase
        p1ind = np.where(self.chainiter < 0)[0]
        p2ind = np.where(self.chainiter >= 0)[0]

        if p1ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise, wvpvs = self.get_weightedvalues(
                pind=p1ind, finaliter=0)
            self.p1models = wmodels  # p1 = phase one
            self.p1misfits = wmisfits
            self.p1likes = wlikes
            self.p1noise = wnoise
            self.p1vpvs = wvpvs
            self.p1temperatures = self.temperatures[:self.iter_phase1]
            if len(self.p1temperatures)!=len(self.p1vpvs):
                raise Exception(len(self.p1temperatures),len(self.p1vpvs))

        if p2ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise, wvpvs = self.get_weightedvalues(
                pind=p2ind, finaliter=self.iiter)
            self.p2models = wmodels  # p2 = phase two
            self.p2misfits = wmisfits
            self.p2likes = wlikes
            self.p2noise = wnoise
            self.p2vpvs = wvpvs
            self.p2temperatures = self.temperatures[self.iter_phase1:]
            if len(self.p2temperatures)!=len(self.p2likes):
                raise Exception(len(self.p2temperatures),len(self.p2likes))

        accmodels = float(self.p2likes.size)  # accepted models in p2 phase
        maxmodels = float(self.initparams['maxmodels'])  # for saving
        self.thinning = int(np.ceil(accmodels / maxmodels))
        self.save_finalmodels()

        logger.debug('time for inversion: %.2f s' % runtime)

    def get_weightedvalues(self, pind, finaliter):
        """
        Models will get repeated (weighted).

        Each iteration, if there was no model proposal accepted, the current
        model gets repeated once more. This weight is based on self.chainiter,
        which documents the iteration of the last accepted model."""
        pmodels = self.chainmodels[pind]  # p = phase (1 or 2)
        pmisfits = self.chainmisfits[pind]
        plikes = self.chainlikes[pind]
        pnoise = self.chainnoise[pind]
        pvpvs = self.chainvpvs[pind]
        pweights = np.diff(np.concatenate((self.chainiter[pind], [finaliter])))

        wmodels, wlikes, wmisfits, wnoise, wvpvs = ModelMatrix.get_weightedvalues(
            pweights, models=pmodels, likes=plikes, misfits=pmisfits,
            noiseparams=pnoise, vpvs=pvpvs)
        return wmodels, wlikes, wmisfits, wnoise, wvpvs

    def save_finalmodels(self):
        """Save chainmodels as pkl file"""
        savepath = op.join(self.initparams['savepath'], 'data')
        names = ['models', 'likes', 'misfits', 'noise', 'vpvs','temperatures']

        # phase 1 -- burnin
        try:
            for i, data in enumerate([self.p1models, self.p1likes,
                                     self.p1misfits, self.p1noise,
                                     self.p1vpvs, self.p1temperatures]):
                outfile = op.join(savepath, 'c%.3d_p1%s' % (self.chainidx, names[i]))
                np.save(outfile, data[::self.thinning])
        except:
            logger.info('No burnin models accepted.')

        # phase 2 -- main / posterior phase
        try:
            for i, data in enumerate([self.p2models, self.p2likes,
                                     self.p2misfits, self.p2noise,
                                     self.p2vpvs, self.p2temperatures]):
                outfile = op.join(savepath, 'c%.3d_p2%s' % (self.chainidx, names[i]))
                np.save(outfile, data[::self.thinning])

            logger.info('> Saving %d models (main phase).' % len(data[::self.thinning]))
        except:
            logger.info('No main phase models accepted.')

    def _init_fixedmodel(self,z,vp,vs):
        self.fixmod_z = z
        self.fixmod_vp = vp
        self.fixmod_vs = vs
        self.fixmod_vs_func = interp1d(self.fixmod_z,self.fixmod_vs,bounds_error=False,fill_value='extrapolate')
        self.fixedvelmodel = True
        imodel = np.hstack((vs,z,np.ones(len(z))*1e-5,np.zeros(len(z))))
        _, _, self.fixmod_h, _, _ = Model.get_vp_vs_h(imodel, self.currentvpvs, self.mantle)
        imodel, vp, vs, h, psi2amp, psi2azi = self._get_fixedmodel_params(self.currentmodel,self.currentvpvs,self.mantle)
        self.targets.evaluate(h=h, vp=vp, vs=vs, noise=self.currentnoise,
                              psi2amp=psi2amp,psi2azi=psi2azi)
        self.accept_as_currentmodel(imodel, self.currentnoise, self.currentvpvs)
        #self.modifications.remove("aniso_birth")
        #self.modifications.remove("aniso_death") # included in normal birth/death
        self.modifications.remove("vsmod") # no vs update needed anymore
        #self.modifications.remove("birth")
        #self.modifications.remove("death")
        #self.modifications.remove("zvmod")
        if self.chainidx==1:
            print("fixed velocity model, only these modifications are included:",self.modifications)

    def _get_fixedmodel_params(self,proposalmodel,proposalvpvs,mantle):
        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(proposalmodel, proposalvpvs, mantle)
        depth = np.append(0,np.repeat(np.cumsum(h),2)[:-1])
        depth[-1] = depth[-2]*1.1 # halfspace
        func = interp1d(depth,np.repeat(psi2amp,2),kind='nearest',bounds_error=False,fill_value='extrapolate')
        psi2amp_resampled = func(self.fixmod_z)
        func = interp1d(depth,np.repeat(psi2azi,2),kind='nearest',bounds_error=False,fill_value='extrapolate')
        psi2azi_resampled = func(self.fixmod_z)
        n = int(len(proposalmodel)/4)
        zi = proposalmodel[n:2*n]
        proposalmodel[:n] = self.fixmod_vs_func(zi)
        return proposalmodel,self.fixmod_vp, self.fixmod_vs, self.fixmod_h, psi2amp_resampled, psi2azi_resampled


