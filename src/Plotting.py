# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
import glob
import logging
import numpy as np
import os.path as op
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from BayHunter import utils
from BayHunter import Targets
from BayHunter import Model, ModelMatrix


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rstate = np.random.RandomState(333)


def vs_round(vs):
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor


def tryexcept(func):
    def wrapper_tryexcept(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            print('* %s: Plotting was not possible\nErrorMessage: %s'
                  % (func.__name__, e))
            return None
    return wrapper_tryexcept


class PlotFromStorage(object):
    """
    Plot and Save from storage (files).
    No chain object is necessary.

    """
    def __init__(self, configfile):
        condict = self.read_config(configfile)
        self.targets = condict['targets']
        self.ntargets = len(self.targets)
        self.refs = condict['targetrefs'] + ['joint']
        self.priors = condict['priors']
        self.initparams = condict['initparams']

        self.datapath = op.dirname(configfile)
        self.figpath = self.datapath.replace('data', '')
        print('Current data path: %s' % self.datapath)

        self.init_filelists()
        self.init_outlierlist()

        self.mantle = self.priors.get('mantle', None)

        self.refmodel = {'model': None,
                         'nlays': None,
                         'noise': None,
                         'vpvs': None,
                         'anisomod': None}

    def read_config(self, configfile):
        return utils.read_config(configfile)

    def savefig(self, fig, filename, dpi=200):
        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight", dpi=dpi)
            plt.close('all')

    def init_outlierlist(self):
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            self.outliers = np.loadtxt(outlierfile, usecols=[0], dtype=int)
            print('Outlier chains from file: %d' % self.outliers.size)
        else:
            print('Outlier chains from file: None')
            self.outliers = np.zeros(0)

    def init_filelists(self):
        filetypes = ['models', 'likes', 'misfits', 'noise', 'vpvs','temperatures']
        filepattern = op.join(self.datapath, 'c???_p%d%s.npy')
        files = []
        size = []

        for ftype in filetypes:
            p1files = sorted(glob.glob(filepattern % (1, ftype)))
            p2files = sorted(glob.glob(filepattern % (2, ftype)))
            files.append([p1files, p2files])
            size.append(len(p1files) + len(p2files))

        if len(set(size)) == 1:
            self.modfiles, self.likefiles, self.misfiles, self.noisefiles, \
                self.vpvsfiles, self.temperaturefiles = files
        else:
            logger.info('You are missing files. Please check ' +
                        '"%s" for completeness.' % self.datapath)
            logger.info('(filetype, number): ' + str(zip(filetypes, size)))

    def get_outliers(self, dev):
        """Detect outlier chains.

        The median likelihood from each chain (main phase) is computed.
        Relatively to the most converged chain, outliers are declared.
        Chains with a deviation of likelihood of dev % are declared outliers.

        Chose dev based on actual results.
        """
        nchains = len(self.likefiles[1])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan

        for i, likefile in enumerate(self.likefiles[1]):
            cidx, _, _ = self._return_c_p_t(likefile)
            chainlikes = np.load(likefile)
            chainmedian = np.median(chainlikes)

            chainidxs[i] = cidx
            chainmedians[i] = chainmedian

        maxlike = np.max(chainmedians)  # best chain average

        # scores must be smaller 1
        if maxlike > 0:
            scores = chainmedians / maxlike
        elif maxlike < 0:
            scores = maxlike / chainmedians

        outliers = chainidxs[np.where(((1-scores) > dev))]
        outscores = 1 - scores[np.where(((1-scores) > dev))]

        if len(outliers) > 0:
            print('Outlier chains found with following chainindices:\n')
            print(outliers)
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write('# Outlier chainindices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(outliers):
                    f.write('%d\t%.3f\n' % (outlier, outscores[i]))

        return outliers

    def _get_chaininfo(self):
        nmodels = [len(np.load(file)) for file in self.likefiles[1]]
        chainlist = [self._return_c_p_t(file)[0] for file in self.likefiles[1]]
        return chainlist, nmodels

    def save_final_distribution(self, maxmodels=200000, dev=0.05):
        """
        Save the final models from all chains, phase 2.

        As input, all the chain files in self.datapath are used.
        Outlier chains will be detected automatically using % dev. The outlier
        detection is based on the maximum reached (median) likelihood
        by the chains. The other chains are compared to the "best" chain and
        sorted out, if the likelihood deviates more than dev * 100 %.

        > Chose dev based on actual results.

        Maxmodels is the maximum number of models to be saved (.npy).
        The chainmodels are combined to one final distribution file,
        while all models are evenly thinned (thinning happens in SingleChain.py).
        """

        def save_finalmodels(models, likes, misfits, noise, vpvs,):
            """Save chainmodels as pkl file"""
            names = ['models', 'likes', 'misfits', 'noise', 'vpvs']
            print('> Saving posterior distribution.')
            for i, data in enumerate([models, likes, misfits, noise, vpvs]):
                outfile = op.join(self.datapath, 'c_%s' % names[i])
                np.save(outfile, data)
                print(outfile)

        # delete old outlier file if evaluating outliers newly
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            os.remove(outlierfile)

        self.outliers = self.get_outliers(dev=dev)

        # due to the forced acceptance rate, each chain should have accepted
        # a similar amount of models. Therefore, a constant number of models
        # will be considered from each chain (excluding outlier chains), to
        # add up to a collection of maxmodels models.
        nchains = int(len(self.likefiles[1]) - self.outliers.size)
        maxmodels = int(maxmodels)
        mpc = int(maxmodels / nchains)  # models per chain

        # # open matrixes and vectors
        allmisfits = None
        allmodels = None
        alllikes = np.ones(maxmodels) * np.nan
        allnoise = np.ones((maxmodels, self.ntargets*4)) * np.nan
        allvpvs = np.ones(maxmodels) * np.nan
        alltemps = np.ones(maxmodels) * np.nan

        start = 0
        chainidxs, nmodels = self._get_chaininfo()

        for i, cidx in enumerate(chainidxs):
            if cidx in self.outliers:
                continue

            index = np.arange(nmodels[i]).astype(int)
            if nmodels[i] > mpc:
                index = rstate.choice(index, mpc, replace=False)
                index.sort()

            chainfiles = [self.modfiles[1][i], self.misfiles[1][i],
                          self.likefiles[1][i], self.noisefiles[1][i],
                          self.vpvsfiles[1][i], self.temperaturefiles[1][i]]

            for c, chainfile in enumerate(chainfiles):
                _, _, ftype = self._return_c_p_t(chainfile)
                data = np.load(chainfile)[index]

                if c == 0:
                    end = start + len(data)

                if ftype == 'likes':
                    alllikes[start:end] = data

                elif ftype == 'models':
                    if allmodels is None:
                        allmodels = np.ones((maxmodels, data[0].size)) * np.nan

                    allmodels[start:end, :] = data

                elif ftype == 'misfits':
                    if allmisfits is None:
                        allmisfits = np.ones((maxmodels, data[0].size)) * np.nan

                    allmisfits[start:end, :] = data

                elif ftype == 'noise':
                    allnoise[start:end, :] = data

                elif ftype == 'vpvs':
                    allvpvs[start:end] = data

                elif ftype == 'temperatures':
                    alltemps[start:end] = data

            start = end

        # exclude nans
        #allmodels = allmodels[~np.isnan(alllikes)]
        #allmisfits = allmisfits[~np.isnan(alllikes)]
        #allnoise = allnoise[~np.isnan(alllikes)]
        #allvpvs = allvpvs[~np.isnan(alllikes)]
        #alltemps = alltemps[~np.isnan(alllikes)]
        #alllikes = alllikes[~np.isnan(alllikes)]

        # exclude temperatures other than 1
        allmodels = allmodels[alltemps==1.]
        allmisfits = allmisfits[alltemps==1.]
        allnoise = allnoise[alltemps==1.]
        allvpvs = allvpvs[alltemps==1.]
        alllikes = alllikes[alltemps==1.]
        #alltemps = alltemps[alltemps==1.]

        save_finalmodels(allmodels, alllikes, allmisfits, allnoise, allvpvs)

    def _unique_legend(self, handles, labels):
        # if a key is double, the last handle in the row is returned to the key
        legend = OrderedDict(zip(labels, handles))
        return legend.values(), legend.keys()

    def _return_c_p_t(self, filename):
        """Return chainindex, phase number, type of file from filename.
        Only for single chain results.
        """
        c, pt = op.basename(filename).split('.npy')[0].split('_')
        cidx = int(c[1:])
        phase, ftype = pt[:2], pt[2:]

        return cidx, phase, ftype

    def _sort(self, chainidxstring):
        chainidx = int(chainidxstring[1:])
        return chainidx

    def _get_layers(self, models):
        layernumber = np.array([(len(model[~np.isnan(model)]) / 4 - 1)
                                for model in models])
        return layernumber

    @tryexcept
    def plot_refmodel(self, fig, mtype='model', **kwargs):
        if fig is not None and self.refmodel[mtype] is not None:
            if mtype == 'nlays':
                nlays = self.refmodel[mtype]
                fig.axes[0].axvline(nlays, color='red', lw=0.5, alpha=0.7)

            if mtype == 'model':
                dep, vs = self.refmodel['model']
                assert len(dep) == len(vs)
                fig.axes[0].plot(vs, dep, **kwargs)
                if len(fig.axes) == 2:
                    deps = np.unique(dep)
                    for d in deps:
                        fig.axes[1].axhline(d, **kwargs)

            if mtype == 'anisomod':
                dep, anisoamp, anisoazi = self.refmodel['anisomod']
                assert len(dep) == len(anisoamp)
                fig.axes[0].plot(anisoamp*100, dep, **kwargs)
                p2am = np.split(anisoazi,np.where(np.abs(np.diff(anisoazi))>np.pi/2.)[0]+1)
                dp = np.split(dep,np.where(np.abs(np.diff(anisoazi))>np.pi/2.)[0]+1)
                for i in range(len(p2am)):
                    fig.axes[1].plot(p2am[i]/np.pi*180, dp[i], **kwargs)

            if mtype == 'noise':
                noise = self.refmodel[mtype]
                for i in range(len(noise)):
                    fig.axes[i].axvline(
                        noise[i], color='red', lw=0.5, alpha=0.7)

            if mtype == 'vpvs':
                vpvs = self.refmodel[mtype]
                fig.axes[0].axvline(vpvs, color='red', lw=0.5, alpha=0.7)
        return fig

# Plot values per iteration.

    def _plot_iitervalues(self, files, ax, layer=0, misfit=0, noise=0, temperature=0, ind=-1):

        unifiles = set([f.replace('p1', 'p2') for f in files])
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, len(unifiles)))

        xmin = -self.initparams['iter_burnin']
        xmax = self.initparams['iter_main']

        files.sort()
        n = 0
        for i, file in enumerate(files):
            phase = int(op.basename(file).split('_p')[1][0])
            alpha = (0.4 if phase==1 else 0.7)
            ls = ('-' if phase==1 else '-')
            lw = (0.5 if phase==1 else 0.8)
            chainidx, _, ftype = self._return_c_p_t(file)
            color = color_list[n]

            data = np.load(file)
            try:
                temperatures = np.load(file.replace(ftype,"temperatures"))
            except:
                temperatures = np.ones(len(data))
            if layer:
                data = self._get_layers(data)
            if misfit or noise:
                data = data.T[ind]

            iters = (np.linspace(xmin, 0, data.size) if phase==1 else
                     np.linspace(0, xmax, data.size))
            if temperature:
                label = 'c%d (#T1:%d)' %(chainidx,len(data[data==1.]))
            else:
                label = 'c%d' % (chainidx)

            if not temperature and not (temperatures==1.).all():
                points = np.array([iters,data]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                linecolors = [color if temperatures[i]==1. else 'lightgrey' 
                              for i in range(len(data))]
                lc = LineCollection(segments, colors=linecolors,
                                    linewidths=lw, alpha=alpha,zorder=-1)
                ax.add_collection(lc)
                ax.plot([], [], color=color,
                        ls=ls, lw=lw, alpha=alpha,
                        label=label if phase==2 else '')
                ax.set_rasterization_zorder(0)
            else:
                ax.plot(iters, data, color=color,
                        ls=ls, lw=lw, alpha=alpha,
                        label=label if phase==2 else '')
                
            if phase == 2:
                if n == 0:
                    datamax = data.max()
                    datamin = data.min()
                else:
                    datamax = np.max([datamax, data.max()])
                    datamin = np.min([datamin, data.min()])
                n += 1

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(datamin*0.95, datamax*1.05)
        ax.axvline(0, color='k', ls=':', alpha=0.7)

        (abs(xmin) + xmax)
        center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]) / (abs(xmin) + xmax)
        for i, text in enumerate(['Burn-in phase', 'Exploration phase']):
            ax.text(center[i], 0.97, text,
                    fontsize=12, color='k',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.set_xlabel('# Iteration')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    @tryexcept
    def plot_iitermisfits(self, nchains=6, ind=-1):
        files = self.misfiles[0][:nchains] + self.misfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, misfit=True, ind=ind)
        ax.set_ylabel('%s misfit' % self.refs[ind])
        return fig

    @tryexcept
    def plot_iiterlikes(self, nchains=6):
        files = self.likefiles[0][:nchains] + self.likefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Likelihood')
        return fig

    @tryexcept
    def plot_iiternoise(self, nchains=6, ind=-1):
        """
        nind = noiseindex, meaning:
        old logic (before introduction of azimuthal anisotropy):
        0: 'rfnoise_corr'  # should be const, if gauss
        1: 'rfnoise_sigma'
        2: 'swdnoise_corr'  # should be 0
        3: 'swdnoise_sigma'
        new logic:
        0: noise_corr target 1
        1: noise_sigma target 1
        2: noise_sigma aa_c1 target1
        3: noise_sigma aa_c2 target1
        4: noise_corr target 2
        5: noise_sigma target 2
        6: and so on...
        # dependent on number and order of targets.
        """
        files = self.noisefiles[0][:nchains] + self.noisefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, noise=True, ind=ind)

        parameter = np.concatenate(
            [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref, '$\sigma~c_1$ (%s)' % ref, '$\sigma~c_2$ (%s)' % ref] for ref in self.refs[:-1]])
        ax.set_ylabel(parameter[ind])
        return fig

    @tryexcept
    def plot_iiternlayers(self, nchains=6):
        files = self.modfiles[0][:nchains] + self.modfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, layer=True)
        ax.set_ylabel('Number of layers')
        return fig

    @tryexcept
    def plot_iitervpvs(self, nchains=6):
        files = self.vpvsfiles[0][:nchains] + self.vpvsfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Vp / Vs')
        return fig

    @tryexcept
    def plot_iitertemperatures(self, nchains=6):
        files = self.temperaturefiles[0][:nchains] + self.temperaturefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, temperature=True)
        ax.set_ylabel('Temperature')
        return fig

# Posterior distributions as 1D histograms for noise and misfits.
# And as 2D histograms / 1D plot for final velocity-depth models
# Considering weighted models.

    @staticmethod
    def _plot_bestmodels(bestmodels, dep_int=None, misfits=None):
        fig, ax = plt.subplots(figsize=(4.4, 7))

        models = ['mean', 'median', 'stdminmax']
        linecolors = ['green', 'blue', 'red']
        ls = ['-', '--', ':']
        lw = [3, 3, 3]

        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int)

        if misfits is not None:
            vs,uidx = np.unique(singlemodels['vs'],axis=0,return_index=True)

            misfit_min = np.min(misfits)
            misfit_max = np.max(misfits)
            norm = colors.LogNorm(vmin=misfit_min,vmax=misfit_max,clip=True)
            cmap = cm.get_cmap('gray')

            segs = []
            cols = []
            misfits_sub = misfits[uidx]
            for i in misfits_sub.argsort()[::-1]:
                segs.append(np.column_stack((vs[i],singlemodels['dep_int'])))
                cols.append(cmap(norm(misfits_sub[i]),alpha=0.9))
            ln_coll = LineCollection(segs, colors = cols,linewidths=0.5,zorder=-5)  
            ax.add_collection(ln_coll)
            ax.set_rasterization_zorder(0)

        for i, model in enumerate(models):
            vs, dep = singlemodels[model]

            ax.plot(vs.T, dep, color=linecolors[i], label=model,
                    ls=ls[i], lw=lw[i])

        ax.invert_yaxis()
        ax.set_ylabel('Depth in km')
        ax.set_xlabel('$V_S$ in km/s')

        han, lab = ax.get_legend_handles_labels()
        ax.legend(han[:-1], lab[:-1], loc=3)
        return fig, ax

    @staticmethod
    def _plot_bestmodels_aniso(bestmodels, dep_int=None, misfits=None):
        fig, axes = plt.subplots(figsize=(4.4, 7),ncols=2, sharey=True)
        ax1 = axes[0]
        ax2 = axes[1]

        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int)
        psi2amp_mean, dep = singlemodels['psi2amp_mean']
        psi2azi_mean, dep = singlemodels['psi2azi_mean']
        psi2amp_std, dep = singlemodels['psi2amp_std']
        psi2azi_std, dep = singlemodels['psi2azi_std']
        psi2azi_mean[psi2azi_mean<0.] += np.pi
        psi2amp,uidx_amp = np.unique(singlemodels['psi2amp'],axis=0,return_index=True)
        psi2azi,uidx_azi = np.unique(singlemodels['psi2azi'],axis=0,return_index=True)
        psi2azi[psi2azi>=np.pi] -= np.pi

        # Variant 1: plot individual models, colored by their misfit
        #misfit_min = np.min(misfits)
        #misfit_max = np.max(misfits)
        #norm = colors.LogNorm(vmin=misfit_min,vmax=misfit_max,clip=True)
        #cmap = cm.get_cmap('jet_r')
        #
        #segs = []
        #cols = []
        #misfits_amp = misfits[uidx_amp]
        #for i in misfits_amp.argsort()[::-1]:
        #    segs.append(np.column_stack((psi2amp[i]*100,dep)))
        #    cols.append(cmap(norm(misfits_amp[i]),alpha=0.9))
        #ln_coll = LineCollection(segs, colors = cols,linewidths=0.5, zorder=-5)  
        #ax1.add_collection(ln_coll)
        #ax1.set_rasterization_zorder(0)

        # Variant 2: plot histograms
        cols = [(1, 1, 1), (0, 0, 1)] # first color is white, last is red
        cmap = colors.LinearSegmentedColormap.from_list("Custom", cols, N=50)
        h1 = ax1.hist2d(singlemodels['psi2amp'].flatten()*100,np.tile(dep,len(singlemodels['psi2amp'])),bins=[np.linspace(0,np.max(psi2amp)*100,31),dep],cmap=cmap,density=True)
        h1[3].set_clim(0,np.max(h1[0])/1.5)

        misfits_azi = misfits[uidx_azi]
        #ax2.scatter(np.hstack(psi2azi)/np.pi*180,np.tile(dep,len(psi2azi)),c=np.tile(misfits_azi,len(dep)),s=1,cmap=cm.jet_r,zorder=-5)
        valid = singlemodels['psi2amp'].flatten()>0.
        h2 = ax2.hist2d(np.hstack(singlemodels['psi2azi'])[valid]/np.pi*180,np.tile(dep,len(singlemodels['psi2azi']))[valid],bins=[np.linspace(0,180,31),dep],cmap=cmap,density=True)
        h2[3].set_clim(0,np.max(h2[0])/1.5)
        ax2.set_rasterization_zorder(0)

        ax1.plot(psi2amp_mean.T*100, dep, color='black', label='mean', ls='-', lw=2)
        ax1.plot((psi2amp_mean+psi2amp_std).T*100, dep, 'k--', label='std', lw=1.5)
        lowstd = (psi2amp_mean-psi2amp_std).T*100
        lowstd[lowstd<0.] = 0.
        ax1.plot(lowstd, dep, 'k--', lw=1.5)
        ax1.legend(loc=4)
        p2am = np.split(psi2azi_mean,np.where(np.abs(np.diff(psi2azi_mean))>np.pi/2.)[0]+1)
        dp = np.split(dep,np.where(np.abs(np.diff(psi2azi_mean))>np.pi/2.)[0]+1)
        for i in range(len(p2am)):
            ax2.plot(p2am[i]/np.pi*180, dp[i], color='black', lw=2)
        stdmax = psi2azi_std>=0.88 # maximum possible std: np.std(np.random.uniform(-np.pi/2.,np.pi/2.,10000)) ~ 0.9
        cdict = {True: "red",False: "black"}
        highstd = psi2azi_mean+psi2azi_std
        highstd[highstd>np.pi] -= np.pi
        split = np.where((np.abs(np.diff(highstd))>np.pi/2.)+(np.diff(stdmax)>0))[0]+1
        p2am = np.split(highstd,split)
        dp = np.split(dep,split)
        smax = np.split(stdmax,split)
        for i in range(len(p2am)):
            ax2.plot(p2am[i][~smax[i]]/np.pi*180, dp[i][~smax[i]], color='black', ls='--', lw=1.5) # std
            ax2.plot(p2am[i][smax[i]]/np.pi*180, dp[i][smax[i]], color='red', ls='--', lw=1.5) # std (maximized)
        lowstd = psi2azi_mean-psi2azi_std
        lowstd[lowstd<0.] += np.pi
        split = np.where((np.abs(np.diff(lowstd))>np.pi/2.)+(np.diff(stdmax)>0))[0]+1
        p2am = np.split(lowstd,split)
        dp = np.split(dep,split)
        smax = np.split(stdmax,split)
        for i in range(len(p2am)):
            ax2.plot(p2am[i][~smax[i]]/np.pi*180, dp[i][~smax[i]], color='black', ls='--', lw=1.5)
            ax2.plot(p2am[i][smax[i]]/np.pi*180, dp[i][smax[i]], color='red', ls='--', lw=1.5)
        ax2.plot([],[],color='black',ls='-',lw=2,label='mean')
        ax2.plot([],[],color='black',ls='--',lw=1.5,label='std')
        ax2.plot([],[],color='red',ls='--',lw=1.5,label='std (maximized)')
        ax2.legend(loc=4)

        ax1.invert_yaxis()
        ax1.set_ylabel('Depth in km')
        ax1.set_xlabel('Aniso amplitude [%]')
        ax2.set_xlabel('Aniso azimuth (math.) [deg]')
        ax2.set_xticks([0,90,180])

        return fig, axes

    @staticmethod
    def _plot_bestmodels_hist(models, dep_int=None):
        """
        2D histogram with 30 vs cells and 50 depth cells.
        As plot depth is limited to 100 km, each depth cell is a 2 km.

        pinterf is the number of interfaces to be plot (derived from gradient)
        """
        if dep_int is None:
            dep_int = np.linspace(0, 100, 201)  # interppolate depth to 0.5 km.
            # bins for 2d histogram
            depbins = np.linspace(0, 100, 101)  # 1 km bins
        else:
            maxdepth = int(np.ceil(dep_int.max()))
            interp = dep_int[1] - dep_int[0]
            dep_int = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
            depbins = np.arange(0, maxdepth + 2*interp, interp)  # interp km bins
            # nbin = np.arange(0, maxdepth + interp, interp)  # interp km bins

        # get interfaces, #first
        models2 = ModelMatrix._replace_zvnoi_h(models)
        models2 = np.array([model[~np.isnan(model)] for model in models2],dtype='object')
        yinterf = np.array([np.cumsum(model[int(model.size/4):-1])
                            for model in models2],dtype='object')
        yinterf = np.concatenate(yinterf)

        vss_int, psi2amps_int, psi2azis_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)
        singlemodels = ModelMatrix.get_singlemodels(models, dep_int=depbins)

        vss_flatten = vss_int.flatten()
        vsinterval = 0.025  # km/s, 0.025 is assumption for vs_round
        # vsbins = int((vss_flatten.max() - vss_flatten.min()) / vsinterval)
        vs_histmin = vs_round(vss_flatten.min())-2*vsinterval
        vs_histmax = vs_round(vss_flatten.max())+3*vsinterval
        vsbins = np.arange(vs_histmin, vs_histmax, vsinterval) # some buffer

        # initiate plot
        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},
                                 sharey=True, figsize=(5, 6.5))
        fig.subplots_adjust(wspace=0.05)

        data2d, xedges, yedges = np.histogram2d(vss_flatten, deps_int.flatten(),
                                				bins=(vsbins, depbins))

        axes[0].imshow(data2d.T, extent=(xedges[0], xedges[-1],
        							     yedges[0], yedges[-1]),
        			   origin='lower',
        			   vmax=len(models), aspect='auto')

        # plot mean / modes
        # colors = ['green', 'white']
        # for c, choice in enumerate(['mean', 'mode']):
        colors = ['white']
        for c, choice in enumerate(['mode']):
            vs, dep = singlemodels[choice]
            color = colors[c]
            axes[0].plot(vs, dep, color=color, lw=1, alpha=0.9, label=choice)

        vs_mode, dep_mode = singlemodels['mode']
        axes[0].legend(loc=3)

        # histogram for interfaces
        data = axes[1].hist(yinterf, bins=depbins, orientation='horizontal',
                            color='lightgray', alpha=0.7,
                            edgecolor='k')
        bins, lay_bin, _ = np.array(data,dtype='object').T
        center_lay = (lay_bin[:-1] + lay_bin[1:]) / 2.

        axes[0].set_ylabel('Depth in km')
        axes[0].set_xlabel('$V_S$ in km/s')

        axes[0].invert_yaxis()

        axes[0].set_title('%d models' % len(models))
        axes[1].set_xticks([])
        return fig, axes

    def _get_posterior_data(self, data, final, chainidx=0):
        if final:
            filetempl = op.join(self.datapath, 'c_%s.npy')
        else:
            filetempl = op.join(self.datapath, 'c%.3d_p2%s.npy' % (chainidx, '%s'))

        outarrays = []
        for dataset in data:
            datafile = filetempl % dataset
            p2data = np.load(datafile)
            try: # only necessary if not final
                temps = np.load(datafile.replace(dataset,"temperatures"))
                p2data = p2data[temps==1.]
            except:
                pass
            outarrays.append(p2data)

        return outarrays

    def _plot_posterior_distribution(self, data, bins, formatter='%.2f', ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5, 3))

        count, bins, _ = ax.hist(data, bins=bins, color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)
        cbins = (bins[:-1] + bins[1:]) / 2.
        mode = cbins[np.argmax(count)]
        median = np.median(data)

        if formatter is not None:
            text = 'median: %s' % formatter % median
            ax.text(0.97, 0.97, text,
                    fontsize=9, color='k',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.axvline(median, color='k', ls=':', lw=1)
        
        # xticks = np.array(ax.get_xticks())
        # ax.set_xticklabels(xticks, fontsize=8)
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    @tryexcept
    def plot_posterior_likes(self, final=True, chainidx=0):
        likes, = self._get_posterior_data(['likes'], final, chainidx)
        bins = 20
        formatter = '%d'

        ax = self._plot_posterior_distribution(likes, bins, formatter)
        ax.set_xlabel('Likelihood')
        return ax.figure

    @tryexcept
    def plot_posterior_misfits(self, final=True, chainidx=0):
        misfits, = self._get_posterior_data(['misfits'], final, chainidx)

        datasets = [misfit for misfit in misfits.T]
        datasets = datasets[:-1]  # excluding joint misfit
        bins = 20
        formatter = '%.2f'

        fig, axes = plt.subplots(1, len(datasets), figsize=(3.5*len(datasets), 3))
        for i, data in enumerate(datasets):
            axes[i] = self._plot_posterior_distribution(data, bins, formatter, ax=axes[i])
            axes[i].set_xlabel('RMS misfit (%s)' % self.refs[i])

        return fig

    @tryexcept
    def plot_posterior_nlayers(self, final=True, chainidx=0):

        models, = self._get_posterior_data(['models'], final, chainidx)

        # get interfaces
        models = np.array([model[~np.isnan(model)] for model in models],dtype='object')
        layers = np.array([(model.size/4 - 1) for model in models],dtype='object')

        bins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatter = '%d'
        ax = self._plot_posterior_distribution(layers, bins, formatter)

        xticks = np.arange(layers.min(), layers.max()+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Number of layers')
        return ax.figure

    @tryexcept
    def plot_posterior_vpvs(self, final=True, chainidx=0):
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)
        bins = 20
        formatter = '%.2f'

        ax = self._plot_posterior_distribution(vpvs, bins, formatter)
        ax.set_xlabel('$V_P$ / $V_S$')
        return ax.figure

    @tryexcept
    def plot_posterior_noise(self, final=True, chainidx=0):
        noise, = self._get_posterior_data(['noise'], final, chainidx)
        label = np.concatenate([['correlation (%s)' % ref, '$\sigma_{iso}$ (%s)' % ref, '$\sigma_{C1}$ (%s)' % ref, '$\sigma_{C2}$ (%s)' % ref]
                               for ref in self.refs[:-1]])

        pars = int(len(noise.T)/4)
        fig, axes = plt.subplots(pars, 4, figsize=(12, 3*pars))
        fig.subplots_adjust(hspace=0.2)
        
        for i, data in enumerate(noise.T):
            if self.ntargets > 1:
                ax = axes[int(i/4)][i % 4]
            else:
                ax = axes[i % 4]

            if np.all(np.isnan(data)):
                ax.set_visible(False)
            elif np.std(data) == 0:  # constant during inversion
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                bins = 20
                formatter = '%.4f'
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
            ax.set_xlabel(label[i])
        return fig

    @tryexcept
    def plot_posterior_others(self, final=True, chainidx=0):
        likes, = self._get_posterior_data(['likes'], final, chainidx)

        misfits, = self._get_posterior_data(['misfits'], final, chainidx)
        misfits = misfits.T[-1]  # only joint misfit
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        models, = self._get_posterior_data(['models'], final, chainidx)
        models = np.array([model[~np.isnan(model)] for model in models])
        layers = np.array([(model.size/4 - 1) for model in models])
        nbins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatters = ['%d', '%.2f', '%.2f', '%d']
        nbins = [20, 20, 20, nbins]
        labels = ['Likelihood', 'Joint misfit', '$V_P$ / $V_S$', 'Number of layers']

        fig, axes = plt.subplots(2, 2, figsize=(7, 6))
        axes = axes.flatten()
        for i, data in enumerate([likes, misfits, vpvs, layers]):
            ax = axes[i]

            if i == 2 and np.std(data) == 0:  # constant vpvs
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                formatter = formatters[i]
                bins = nbins[i]
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)

                if i == 3:
                    xticks = np.arange(layers.min(), layers.max()+1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)

            ax.set_xlabel(labels[i])
        return ax.figure

    @tryexcept
    def plot_posterior_models1d(self, final=True, chainidx=0, depint=1):
        """depint is the depth interpolation used for binning. Default=1km."""
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        models, = self._get_posterior_data(['models'], final, chainidx)
        misfits, = self._get_posterior_data(['misfits'], final, chainidx)
        datasets = [misfit for misfit in misfits.T]
        misfits = datasets[-1]  # only joint misfit

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        fig1, ax = self._plot_bestmodels(models, dep_int, misfits)
        # ax.set_xlim(self.priors['vs'])
        ax.set_yticks(np.arange(0,300,10))
        ax.set_ylim(self.priors['z'][::-1])
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        ax.set_title('%d models from %d chains' % (len(models), nchains))

        if self.initparams['azimuthal_anisotropy']:
            fig2, axes = self._plot_bestmodels_aniso(models, dep_int, misfits)
            for ax in axes:
                ax.set_yticks(np.arange(0,300,10))
                ax.set_ylim(self.priors['z'][::-1])
                ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        else:
            fig2 = plt.figure()

        return fig1,fig2

    @tryexcept
    def plot_posterior_models2d(self, final=True, chainidx=0, depint=1):
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)

        fig, axes = self._plot_bestmodels_hist(models, dep_int)
        # axes[0].set_xlim(self.priors['vs'])
        axes[0].set_ylim(self.priors['z'][::-1])
        axes[0].set_title('%d models from %d chains' % (len(models), nchains))
        return fig


# Plot moho depth - crustal vs tradeoff

    @tryexcept
    def plot_moho_crustvel_tradeoff(self, moho=None, mohovs=None, refmodel=None):
        models, vpvs = self._get_posterior_data(['models', 'vpvs'], final=True)

        if moho is None:
            moho = self.priors['z']
        if mohovs is None:
            mohovs = 4.2  # km/s

        mohos = np.zeros(len(models)) * np.nan
        vscrust = np.zeros(len(models)) * np.nan
        vslastlayer = np.zeros(len(models)) * np.nan
        vsjumps = np.zeros(len(models)) * np.nan

        for i, model in enumerate(models):
            thisvpvs = vpvs[i]
            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(model, thisvpvs, self.mantle)
            # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
            # ifaces, vs = cdepth[1::2], cvs[::2]   # interfaces, vs
            ifaces = np.cumsum(h)
            vsstep = np.diff(vs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            if len(mohoidxs) == 0:
                continue

            # mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            mohoidxs = mohoidxs.flatten()

            mohoidxs_vs = np.where((vs > mohovs))[0]-1
            if len(mohoidxs_vs) == 0:
                continue

            mohoidx = np.intersect1d(mohoidxs, mohoidxs_vs)
            if len(mohoidx) == 0:
                continue
            mohoidx = mohoidx[0]
            # ------

            thismoho = ifaces[mohoidx]
            crustmean = np.sum(vs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            lastvs = vs[mohoidx]
            vsjump = vsstep[mohoidx]

            mohos[i] = thismoho
            vscrust[i] = crustmean
            vslastlayer[i] = lastvs
            vsjumps[i] = vsjump

        # exclude nan values
        mohos = mohos[~np.isnan(vsjumps)]
        vscrust = vscrust[~np.isnan(vsjumps)]
        vslastlayer = vslastlayer[~np.isnan(vsjumps)]
        vsjumps = vsjumps[~np.isnan(vsjumps)]

        fig, ax = plt.subplots(2, 4, figsize=(11, 6))
        fig.subplots_adjust(hspace=0.05)
        fig.subplots_adjust(wspace=0.05)

        labels = ['$V_S$ last crustal layer', '$V_S$ crustal mean', '$V_S$ increase']
        bins = 50

        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                histdata = ax[0][n].hist(xdata, bins=bins,
                                         color='darkblue', alpha=0.7,
                                         edgecolor='white', linewidth=0.4)

                median = np.median(xdata)
                ax[0][n].axvline(median, color='k', ls='--', lw=1.2, alpha=1)
                stats = 'median:\n%.2f km/s' % median
                ax[0][n].text(0.97, 0.97, stats,
                              fontsize=9, color='k',
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform=ax[0][n].transAxes)
            except:
                pass

        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                ax[1][n].set_xlabel(labels[n])

                data = ax[1][n].hist2d(xdata, mohos, bins=bins)
                data2d, xedges, yedges, _ = np.array(data).T

                xi, yi = np.unravel_index(data2d.argmax(), data2d.shape)
                x_mode = ((xedges[:-1] + xedges[1:]) / 2.)[xi]
                y_mode = ((yedges[:-1] + yedges[1:]) / 2.)[yi]

                ax[1][n].axhline(y_mode, color='white', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(x_mode, color='white', ls='--', lw=0.5, alpha=0.7)

                xmin, xmax = ax[1][n].get_xlim()
                ax[0][n].set_xlim([xmin, xmax])
            except:
                pass

            ax[0][n].set_yticks([])
            ax[0][n].set_yticklabels([], visible=False)
            ax[0][n].set_xticklabels([], visible=False)

        ax[1][1].set_yticklabels([], visible=False)
        ax[1][2].set_yticklabels([], visible=False)
        ax[1][3].set_yticklabels([], visible=False)
        ax[1][0].set_ylabel('Moho depth in km')

        # plot moho 1d histogram
        histdata = ax[1][3].hist(mohos, bins=bins, orientation='horizontal',
                                 color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)

        median = np.median(mohos)
        std = np.std(mohos)
        print('moho: %.4f +- %.4f km' % (median, std))
        ax[1][3].axhline(median, color='k', ls='--', lw=1.2, alpha=1)
        stats = 'median:\n%.2f km' % median
        ax[1][3].text(0.97, 0.97, stats,
                      fontsize=9, color='k',
                      horizontalalignment='right',
                      verticalalignment='top',
                      transform=ax[1][3].transAxes)
        ymin, ymax = ax[1][0].get_ylim()
        # ymin, ymax = median - 4*std, median + 4*std
        ax[1][0].set_ylim(ymin, ymax)
        ax[1][1].set_ylim(ymin, ymax)
        ax[1][2].set_ylim(ymin, ymax)
        ax[1][3].set_ylim(ymin, ymax)

        ax[1][3].set_xticklabels([], visible=False)
        ax[1][3].set_yticks([])
        ax[1][3].set_yticklabels([], visible=False)
        ax[0][3].axis('off')

        if refmodel is not None:
            dep, vs = refmodel
            h = (dep[1:] - dep[:-1])[::2]
            ifaces, lvs = dep[1::2], vs[::2]

            vsstep = np.diff(lvs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            truemoho = ifaces[mohoidx]
            truecrust = np.sum(lvs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            truevslast = lvs[mohoidx]
            truevsjump = vsstep[mohoidx]

            for n, xdata in enumerate([truevslast, truecrust, truevsjump]):
                ax[1][n].axhline(truemoho, color='red', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(xdata, color='red', ls='--', lw=0.5, alpha=0.7)

        return fig

# Plot current models and data fits. also plot best data fit incl. model.

    @tryexcept
    def plot_currentmodels(self, nchains):
        """Return fig.

        Plots the first nchains chains, no matter of outlier status.
        """
        fig, ax = plt.subplots(figsize=(4, 6.5))

        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            color = color_list[i]
            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)

            label = 'c%d / %d' % (chainidx, vs.size-1)
            ax.plot(cvs, cdepth, color=color, ls='-', lw=0.8,
                    alpha=0.7, label=label)

        ax.invert_yaxis()
        ax.set_xlabel('$V_S$ in km/s')
        ax.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax.set_ylim(self.priors['z'][::-1])
        ax.set_title('Current models')
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig

    #@tryexcept
    def plot_currentdatafits(self, nchains):
        """Plot the first nchains chains, no matter of outlier status.
        """
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))
        targets = Targets.JointTarget(targets=self.targets)

        fig, ax = targets.plot_obsdata(mod=False)
        if type(ax) != type(np.array([])):
            ax = np.array([ax],dtype='object')

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            color = color_list[i]
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            rho = vp * 0.32 + 0.77
            c1 = psi2amp * np.cos(2*psi2azi)
            c2 = psi2amp * np.sin(2*psi2azi)

            jmisfit = 0
            for n, target in enumerate(targets.targets):
                #xmod, ymod = target.moddata.plugin.run_model(
                #    h=h, vp=vp, vs=vs, rho=rho)
                target.moddata.calc_synth(h, vp, vs, rho=rho, c1=c1, c2=c2)
                xmod = target.moddata.x
                ymod = target.moddata.y
                
                yobs = target.obsdata.y
                misfit = target.valuation.get_rms(yobs, ymod)
                jmisfit += misfit

                if target.azimuthal_anisotropic:
                    aniso_amp = target.moddata.aa_amp*100
                    aniso_azi = target.moddata.aa_ang
                else:
                    aniso_amp = None

                label = ''
                if len(targets.targets) > 1:
                    if ((len(targets.targets) - 1) - n) < 1e-2:
                        label = 'c%d / %.3f' % (chainidx, jmisfit)
                    ax[n][0].plot(xmod, ymod, color=color, alpha=0.7, lw=0.8,
                                  label=label)
                    if aniso_amp is not None:
                        ax[n][1].plot(xmod, aniso_amp, color=color, alpha=0.7, lw=0.8,
                                      label=label)
                        ax[n][2].plot(xmod, aniso_azi/np.pi*180.,'o', color=color, alpha=0.7,
                                      ms=2,label=label)
                else:
                    label = 'c%d / %.3f' % (chainidx, jmisfit)
                    ax[0].plot(xmod, ymod, color=color, alpha=0.5, lw=0.7,
                               label=label)
                    if len(ax)>1:
                        ax[1].plot(xmod, aniso_amp, color=color, alpha=0.5, lw=0.7,
                                   label=label)
                        ax[2].plot(xmod, aniso_azi/np.pi*180.,'o', color=color, alpha=0.5,
                                   ms=2,label=label)

        if len(targets.targets) > 1:
            ax[0][0].set_title('Current data fits')
            idx = len(targets.targets) - 1
            han, lab = ax[idx][0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0][0].legend().set_visible(False)
        else:
            ax[0].set_title('Current data fits')
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    #@tryexcept
    def plot_meandatafits(self,depint,aziamp=True):
        
        targets = Targets.JointTarget(targets=self.targets)
        models, = self._get_posterior_data(['models'],final=True)

        fig, ax = targets.plot_obsdata(ax=None,mod=False,aziamp=aziamp)
        if type(ax) != type(np.array([])):
            ax = np.array([ax],dtype='object')
        xbounds = []    
        for a in ax.flatten():
            try:
                a.lines[0].set_markersize(5)
            except:
                pass
            a.set_xscale('log')
            xbounds.append(a.get_xbound())

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        singlemodels = ModelMatrix.get_singlemodels(models, dep_int)
        psi2amp_mean, dep = singlemodels['psi2amp_mean']
        psi2azi_mean, dep = singlemodels['psi2azi_mean']
        if psi2amp_mean is None:
            psi2amp_mean = psi2azi_mean = np.zeros(len(dep))
        psi2azi_mean[psi2azi_mean<0.] += np.pi
        vs_mean, dep = singlemodels['mean']

        _,uidx = np.unique(models,axis=0,return_index=True)
        thin = int(np.floor(len(uidx)/500))
        if thin>0:
            uidx = uidx[::thin]
        misfits, = self._get_posterior_data(['misfits'], final=True)
        vpvs, = self._get_posterior_data(['vpvs'], final=True)
        meanvpvs = np.mean(vpvs)
        models = models[uidx]
        misfits = misfits[:,-1][uidx]
        vpvs = vpvs[uidx]

        misfit_min = np.min(misfits)
        misfit_max = np.max(misfits)
        norm = colors.LogNorm(vmin=misfit_min,vmax=misfit_max,clip=True)
        cmap = cm.get_cmap('jet_r')

        segs_phvel = []
        segs_aa1 = []
        segs_aa2 = []
        for n, target in enumerate(self.targets):
            segs_phvel.append([])
            segs_aa1.append([])
            segs_aa2.append([])
        cols = []
        for i in misfits.argsort()[::-1]:
            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(models[i], vpvs[i], self.mantle)
            rho = vp * 0.32 + 0.77
            c1 = psi2amp * np.cos(2*psi2azi)
            c2 = psi2amp * np.sin(2*psi2azi)
            for n, target in enumerate(self.targets):
                target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                                          c1=c1, c2=c2)
                phveliso = target.moddata.y
                segs_phvel[n].append(np.column_stack((target.moddata.x,phveliso)))
                if target.azimuthal_anisotropic and aziamp:
                    anisoamp = target.moddata.aa_amp*100
                    anisoazi = target.moddata.aa_ang
                    segs_aa1[n].append(np.column_stack((target.moddata.x,anisoamp)))
                    segs_aa2[n].append(np.column_stack((target.moddata.x,anisoazi)))
                elif target.azimuthal_anisotropic and not aziamp:
                    segs_aa1[n].append(np.column_stack((target.moddata.x,target.moddata.c1)))
                    segs_aa2[n].append(np.column_stack((target.moddata.x,target.moddata.c2)))
            cols.append(cmap(norm(misfits[i]),alpha=0.9))
        for n, target in enumerate(self.targets):
            if len(ax.shape)>1:
                a = ax[n]
            else:
                a = ax
            ln_coll = LineCollection(segs_phvel[n], colors = cols,linewidths=0.5, zorder=-5)
            a[0].add_collection(ln_coll)
            a[0].set_rasterization_zorder(0)
            if target.azimuthal_anisotropic:
                ln_coll = LineCollection(segs_aa1[n], colors = cols,linewidths=0.5, zorder=-5)
                a[1].add_collection(ln_coll)
                a[1].set_rasterization_zorder(0)
                if aziamp:
                    a[2].scatter(np.vstack(segs_aa2[n])[:,0],np.vstack(segs_aa2[n])[:,1]/np.pi*180,
                                 c=np.tile(misfits,len(target.moddata.x)),s=1,cmap=cm.jet_r,zorder=-5)
                else:
                    ln_coll = LineCollection(segs_aa2[n], colors = cols,linewidths=0.5, zorder=-5)
                    a[2].add_collection(ln_coll)
                a[2].set_rasterization_zorder(0)

        meanmodel = np.hstack((vs_mean,dep,psi2amp_mean,psi2azi_mean))
        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(meanmodel, meanvpvs, self.mantle)
        rho = vp * 0.32 + 0.77
        c1 = psi2amp * np.cos(2*psi2azi)
        c2 = psi2amp * np.sin(2*psi2azi)
        for n, target in enumerate(self.targets):
            target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                                      c1=c1, c2=c2)
            phveliso = target.moddata.y
            if len(ax.shape)>1:
                a = ax[n]
            else:
                a = ax
            a[0].plot(target.moddata.x,phveliso,'k',label='mean')
            if target.azimuthal_anisotropic and aziamp: 
                anisoamp = target.moddata.aa_amp*100
                anisoazi = target.moddata.aa_ang
                a[1].plot(target.moddata.x,anisoamp,'k',label='mean')
                a[2].plot(target.moddata.x,anisoazi/np.pi*180,'ko',label='mean')
            elif target.azimuthal_anisotropic and not aziamp:
                a[1].plot(target.moddata.x,target.moddata.c1,'k',label='mean')
                a[2].plot(target.moddata.x,target.moddata.c2,'k',label='mean')

        if len(targets.targets) > 1:
            ax[0][0].set_title('Data fits')
            idx = len(targets.targets) - 1
            han, lab = ax[idx][0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0][0].legend().set_visible(False)
        else:
            ax[0].set_title('Current data fits')
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)

        for i,a in enumerate(ax.flatten()):
            a.set_xbound(xbounds[i])

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))

        return fig


    @tryexcept
    def plot_bestmodels(self):
        """Return fig.

        Plot the best (fit) models ever discovered per each chain,
        ignoring outliers.
        """
        fig, ax = plt.subplots(figsize=(4, 6.5))

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]

        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)

            ax.plot(cvs, cdepth, color='k', ls='-', lw=0.8, alpha=0.5)

        # label = 'c%d' % thebestchain
        # vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(thebestmodel, thebestvpvs, self.mantle)
        # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
        # ax.plot(cvs, cdepth, color='red', ls='-', lw=1,
        #         alpha=0.8, label=label)

        ax.invert_yaxis()
        ax.set_xlabel('$V_S$ in km/s')
        ax.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax.set_ylim(self.priors['z'][::-1])
        ax.set_title('Best fit models from %d chains' %
                     (len(modfiles)-self.outliers.size))
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        # ax.legend(loc=3)
        return fig

    @tryexcept
    def plot_bestdatafits(self):
        """Plot best data fits from each chain and ever best,
        ignoring outliers."""
        targets = Targets.JointTarget(targets=self.targets)
        fig, ax = targets.plot_obsdata(mod=False)

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]

        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            rho = vp * 0.32 + 0.77
            c1 = psi2amp * np.cos(2*psi2azi)
            c2 = psi2amp * np.sin(2*psi2azi)

            for n, target in enumerate(targets.targets):
                target.moddata.calc_synth(h, vp, vs, rho=rho, c1=c1, c2=c2)
                # only isotropic part
                xmod = target.moddata.x
                ymod = target.moddata.y

                if len(targets.targets) > 1:
                    ax[n].plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)
                else:
                    ax.plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)

        if len(targets.targets) > 1:
            ax[0].set_title('Best data fits from %d chains' %
                            (len(modfiles)-self.outliers.size))
            # idx = len(targets.targets) - 1
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Best data fits from %d chains' %
                         (len(modfiles)-self.outliers.size))
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    @tryexcept
    def plot_rfcorr(self, rf='prf'):
        from BayHunter import SynthObs

        p2models, p2noise, p2misfits, p2vpvs = self._get_posterior_data(
            ['models', 'noise', 'misfits', 'vpvs'], final=True)

        fig, axes = plt.subplots(2, sharex=True, sharey=True)
        ind = self.refs.index(rf)
        best = np.argmin(p2misfits.T[ind])
        model = p2models[best]
        vpvs = p2vpvs[best]

        target = self.targets[ind]
        x, y = target.obsdata.x, target.obsdata.y
        vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(model, vpvs, self.mantle)
        rho = vp * 0.32 + 0.77

        _, ymod = target.moddata.plugin.run_model(
            h=h, vp=vp, vs=vs, rho=rho)
        yobs = target.obsdata.y
        yresiduals = yobs - ymod

        # axes[0].set_title('Residuals [dobs-g(m)] obtained with best fitting model m')
        axes[0].plot(x, yresiduals, color='k', lw=0.7, label='residuals')

        corr, sigma = p2noise[best][4*ind:4*(ind+1)]
        yerr = SynthObs.compute_gaussnoise(y, corr=corr, sigma=sigma)
        # axes[1].set_title('One Realization of random noise from inferred CeRF')
        axes[1].plot(x, yerr, color='k', lw=0.7, label='noise realization')
        axes[1].set_xlabel('Time in s')

        axes[0].legend(loc=4)
        axes[1].legend(loc=4)
        axes[0].grid(color='gray', ls=':', lw=0.5)
        axes[1].grid(color='gray', ls=':', lw=0.5)
        axes[0].set_xlim([x[0], x[-1]])

        return fig

    def merge_pdfs(self):
        from PyPDF2 import PdfFileReader, PdfFileWriter

        outputfile = op.join(self.figpath, 'c_summary.pdf')
        output = PdfFileWriter()
        pdffiles = glob.glob(op.join(self.figpath + os.sep + 'c_*.pdf'))
        pdffiles.sort(key=op.getmtime)

        for pdffile in pdffiles:
            if pdffile == outputfile:
                continue

            document = PdfFileReader(open(pdffile, 'rb'))
            for i in range(document.getNumPages()):
                output.addPage(document.getPage(i))

        with open(outputfile, "wb") as f:
            output.write(f)

    def save_chainplots(self, cidx=0, refmodel=dict(), depint=None, dpi=200):
        """
        Refmodel is a dictionary and must contain plottable values:
        - 'vs' and 'dep' for the vs-depth plots, will be plotted as given
        - 'rfnoise_corr', 'rfnoise_sigma', 'swdnoise_corr', 'swdnoise_sigma' -
        in this order as noise parameter reference in histogram plots
        - 'nlays' number of layers as reference

        Only given values will be plotted.

        - depint is the interpolation only for histogram plotting.
        Default is 1 km. A finer interpolation increases the plotting time.
        """
        self.refmodel.update(refmodel)
        # plot chain specific posterior distributions

        fig5a = self.plot_posterior_misfits(final=False, chainidx=cidx)
        self.savefig(fig5a, 'c%.3d_posterior_misfit.pdf' % cidx, dpi=dpi)

        fig5b = self.plot_posterior_nlayers(final=False, chainidx=cidx)
        self.plot_refmodel(fig5b, 'nlays')
        self.savefig(fig5b, 'c%.3d_posterior_nlayers.pdf' % cidx, dpi=dpi)

        fig5c = self.plot_posterior_noise(final=False, chainidx=cidx)
        self.plot_refmodel(fig5c, 'noise')
        self.savefig(fig5c, 'c%.3d_posterior_noise.pdf' % cidx, dpi=dpi)

        fig5d_a,fig5d_b = self.plot_posterior_models1d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5d_a, 'model', color='k', lw=1)
        self.savefig(fig5d_a, 'c%.3d_posterior_models1d.pdf' % cidx, dpi=dpi)
        self.savefig(fig5d_b, 'c%.3d_posterior_models2d_aniso.pdf' % cidx, dpi=dpi)

        fig5e = self.plot_posterior_models2d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5e, 'model', color='red', lw=0.5, alpha=0.7)
        self.savefig(fig5e, 'c%.3d_posterior_models2d.pdf' % cidx, dpi=dpi)

    def save_plots(self, nchains=5, refmodel=dict(), depint=1, dpi=200):
        """
        Refmodel is a dictionary and must contain plottable values:
        - 'vs' and 'dep' (np.arrays) for the vs-depth plots, will be plotted as given
        - noise parameters, if e.g., inverting for RF and SWD are:
        'rfnoise_corr', 'rfnoise_sigma', 'swdnoise_corr', 'swdnoise_sigma',
        (depends on number of targets, but order must be correlation / sigma)
        - 'nlays' number of layers as reference

        Only given values will be plotted.

        - depint is the interpolation only for histogram plotting.
        Default is 1 km. A finer interpolation increases the plotting time.
        """
        self.refmodel.update(refmodel)

        nchains = np.min([nchains, len(self.likefiles[1])])

        # plot values changing over iteration
        fig1a = self.plot_iiterlikes(nchains=nchains)
        self.savefig(fig1a, 'c_iiter_likes.pdf', dpi=dpi)

        fig1b = self.plot_iitermisfits(nchains=nchains, ind=-1)
        self.savefig(fig1b, 'c_iiter_misfits.pdf', dpi=dpi)

        fig1c = self.plot_iiternlayers(nchains=nchains)
        self.savefig(fig1c, 'c_iiter_nlayers.pdf', dpi=dpi)

        fig1d = self.plot_iitervpvs(nchains=nchains)
        self.savefig(fig1d, 'c_iiter_vpvs.pdf', dpi=dpi)

        for i in range(self.ntargets):
            ind = i * 4 + 1 # ind=0: correlation coeff, ind=1: noise, ind=2: aniso_c1 noise, ind=3: aniso_c2 noise
            fig1e = self.plot_iiternoise(nchains=nchains, ind=ind)
            self.savefig(fig1e, 'c_iiter_noisepar%d.pdf' % ind, dpi=dpi)

        fig1f = self.plot_iitertemperatures(nchains=nchains)
        self.savefig(fig1f, 'c_iiter_temperatures.pdf', dpi=dpi)

        # plot current models and datafit
        fig3a = self.plot_currentmodels(nchains=nchains)
        self.plot_refmodel(fig3a, 'model', color='k', lw=1)
        self.savefig(fig3a, 'c_currentmodels.pdf', dpi=dpi)

        fig3b = self.plot_currentdatafits(nchains=nchains)
        self.savefig(fig3b, 'c_currentdatafits.pdf', dpi=dpi)

        fig4a = self.plot_meandatafits(depint=depint)
        self.savefig(fig4a, 'c_meandatafits_a.pdf', dpi=dpi)

        fig4b = self.plot_meandatafits(depint=depint,aziamp=False)
        self.savefig(fig4b, 'c_meandatafits_b.pdf', dpi=dpi)

        # plot final posterior distributions
        fig2b = self.plot_posterior_nlayers()
        self.plot_refmodel(fig2b, 'nlays')
        self.savefig(fig2b, 'c_posterior_nlayers.pdf', dpi=dpi)

        fig2b = self.plot_posterior_vpvs()
        self.plot_refmodel(fig2b, 'vpvs')
        self.savefig(fig2b, 'c_posterior_vpvs.pdf', dpi=dpi)

        fig2c = self.plot_posterior_noise()
        self.plot_refmodel(fig2c, 'noise')
        self.savefig(fig2c, 'c_posterior_noise.pdf', dpi=dpi)

        fig2d_a,fig2d_b = self.plot_posterior_models1d(depint=depint)
        self.plot_refmodel(fig2d_a, 'model', color='magenta', lw=1)
        self.savefig(fig2d_a, 'c_posterior_models1d.pdf', dpi=dpi)
        self.plot_refmodel(fig2d_b, 'anisomod', color='magenta', lw=1)
        self.savefig(fig2d_b, 'c_posterior_models1d_aniso.pdf', dpi=dpi)
        fig2e = self.plot_posterior_models2d(depint=depint)
        self.plot_refmodel(fig2e, 'model', color='red', lw=0.5, alpha=0.7)
        self.savefig(fig2e, 'c_posterior_models2d.pdf', dpi=dpi)
