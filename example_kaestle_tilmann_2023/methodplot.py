#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:23:44 2022

@author: emanuel
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

from BayHunter import utils
from BayHunter import PlotFromStorage
from BayHunter import ModelMatrix, Model
from BayHunter import Targets

def vs_round(vs):
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor

# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)
mantle = priors['mantle']

# path to folder where the output files are stored
path = "results_test_0/data"

testfile = "rayleigh_profiles/profile_78846_5955_13.500_46.500.txt"

data = np.loadtxt(testfile)
data = data[data[:,0].argsort()]
xray = data[:,0]
yray = data[:,1]
valid = ~np.isnan(yray)*(xray>=2.5)*(xray<=50.)*(data[:,2]<0.3)
veljumps = np.where(np.diff(yray)<-0.2)[0]
if len(veljumps)>0:
    valid[:veljumps[-1]+1] = False
if np.sum(valid)<5:
    raise Exception()
xray = xray[valid]
yray = yray[valid]
yray_err = data[valid,2]
yray_c1 = data[valid,3]
yray_c1err = data[valid,4]
yray_c2 = data[valid,5]
yray_c2err = data[valid,6]

data_lov = np.loadtxt(testfile.replace("rayleigh","love"))
data_lov = data_lov[data_lov[:,0].argsort()]
xlov = data_lov[:,0]
ylov = data_lov[:,1]
valid = ~np.isnan(ylov)*(xlov>=2.5)*(xlov<=50.)*(data_lov[:,2]<0.3)
veljumps = np.where(np.diff(ylov)<-0.2)[0]
if len(veljumps)>0:
    valid[:veljumps[-1]+1] = False
if np.sum(valid)<5:
    raise Exception()
xlov = xlov[valid]
ylov= ylov[valid]

target1 = Targets.RayleighDispersionPhase(xray, yray, yerr=yray_err,
                                          c1=yray_c1, c1err=yray_c1err,
                                          c2=yray_c2, c2err=yray_c2err
                                          )
target2 = Targets.LoveDispersionPhase(xlov, ylov)
targets = [target1]

# #  ---------------------------------------------- Model resaving and plotting
cfile = '%s_config.pkl' % initparams['station']
configfile = op.join(path, cfile)
obj = PlotFromStorage(configfile)

depint = 1
models, = obj._get_posterior_data(["models"],final=True)
vpvs, = obj._get_posterior_data(['vpvs'], final=True)
vpvs_mean = np.mean(vpvs)
misfits, = obj._get_posterior_data(['misfits'], final=True)
noise, = obj._get_posterior_data(['noise'], final=True)
likes, = obj._get_posterior_data(['likes'], final=True)
raystd_mean = np.mean(noise[:,1])
#lovstd_mean = np.mean(noise[:,5])
dep_int = np.arange(obj.priors['z'][0],obj.priors['z'][1] + depint, depint)
model_elements = ['mean', 'median', 'stdminmax']
singlemodels = ModelMatrix.get_singlemodels(models, dep_int)
vs_mean, dep = singlemodels['mean']
vs_median, dep = singlemodels['median']
vs_mode, depi = singlemodels['mode']
vs_mode = np.interp(dep,depi,vs_mode,)
vs_std, dep = singlemodels['stdminmax']
psi2amp_mean, dep = singlemodels['psi2amp_mean']
psi2azi_mean, dep = singlemodels['psi2azi_mean']
psi2azi_mean[psi2azi_mean<0.] += np.pi
psi2amp_std, dep = singlemodels['psi2amp_std']
psi2azi_std, dep = singlemodels['psi2azi_std']
psi2amp,uidx_amp = np.unique(singlemodels['psi2amp'],axis=0,return_index=True)
psi2azi,uidx_azi = np.unique(singlemodels['psi2azi'],axis=0,return_index=True)
psi2azi[psi2azi>=np.pi] -= np.pi

vss_int = None



#%%
labelpad = 0
fig = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(3, 5,width_ratios=[1.7,1.4,1,0.8,0.8],height_ratios=[1,1,1],wspace=0.3,hspace=0.5)

ax1 = fig.add_subplot(gs[0,0])
ax1.set_title("(a)",loc='left',fontsize=14)
_,uidx = np.unique(models,axis=0,return_index=True)
thin = int(np.floor(len(uidx)/500))
if thin>0:
    uidx = uidx[::thin]
modelsU = models[uidx]
misfitsU = misfits[:,-1][uidx]
vpvsU = vpvs[uidx]

misfit_min = np.min(misfitsU)
misfit_max = np.max(misfitsU)
norm = colors.LogNorm(vmin=misfit_min,vmax=misfit_max,clip=True)
cmap = plt.cm..jet_r

for target in targets:
    if target.ref.startswith('l'):
        label = '$d^{obs}_L$'
        color = 'magenta'
        linestyle='solid'
    else:
        label = '$c^{obs}$'
        color = 'k'
        linestyle='solid'
    ax1.errorbar(target.obsdata.x, target.obsdata.y, yerr=target.obsdata.yerr,
                 label=label,color='blue', fmt='o', ms=2, lw=1.5, zorder=1000)    

segs_phvel = []
segs_aa1 = []
segs_aa2 = []
for n, target in enumerate(targets):
    segs_phvel.append([])
    segs_aa1.append([])
    segs_aa2.append([])
cols = []
aziamp = False
for i in misfitsU.argsort()[::-1]:
    vp, vs, h, a2, psi2 = Model.get_vp_vs_h(modelsU[i], vpvsU[i], mantle)
    rho = vp * 0.32 + 0.77
    c1 = a2 * np.cos(2*psi2)
    c2 = a2 * np.sin(2*psi2)
    for n, target in enumerate(targets):
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
    cols.append(cmap(norm(misfitsU[i]),alpha=0.9))
for n, target in enumerate(targets):
    ln_coll = LineCollection(segs_phvel[n], colors = cols,linewidths=0.5, zorder=-5)
    ax1.add_collection(ln_coll)
    ax1.set_rasterization_zorder(0)

meanmodel = np.hstack((vs_mean,dep,psi2amp_mean,psi2azi_mean))
vp, vs, h, a2, psi2 = Model.get_vp_vs_h(meanmodel, vpvs_mean, mantle)
rho = vp * 0.32 + 0.77
c1 = a2 * np.cos(2*psi2)
c2 = a2 * np.sin(2*psi2)
for n, target in enumerate(targets):
    target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                              c1=c1, c2=c2)
    phveliso = target.moddata.y
    ax1.plot(target.moddata.x,phveliso,'--',color='k')
ax1.plot([],[],'r',linewidth=0.5,label='$c^{mod}$')
ax1.plot([],[],'--',color='k',label='mean')
ax1.legend(bbox_to_anchor=(1,1))
ax1.set_xscale('log')
ax1.set_xticks([2.5,4,6,10,20,50])
ax1.set_xticklabels(["2.5","4","6","10","20","50",])
ax1.set_xlabel("$period~[s]$",labelpad=labelpad)
ax1.set_ylabel("$c_0~[km/s]$",labelpad=labelpad)

##########################################################################################################################
ax11 = fig.add_subplot(gs[1,0])
ax11.set_title("(b)",loc='left',fontsize=14)

ax11.errorbar(target.obsdata.x, target.obsdata.c1, yerr=target.obsdata.c1_err,
              label=label,color='blue', fmt='o', ms=2, lw=1.5, zorder=1000)    

segs_phvel = []
segs_aa1 = []
segs_aa2 = []
for n, target in enumerate(targets):
    segs_phvel.append([])
    segs_aa1.append([])
    segs_aa2.append([])
cols = []
aziamp = False
for i in misfitsU.argsort()[::-1]:
    vp, vs, h, a2, psi2 = Model.get_vp_vs_h(modelsU[i], vpvsU[i], mantle)
    rho = vp * 0.32 + 0.77
    c1 = a2 * np.cos(2*psi2)
    c2 = a2 * np.sin(2*psi2)
    for n, target in enumerate(targets):
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
    cols.append(cmap(norm(misfitsU[i]),alpha=0.9))
for n, target in enumerate(targets):
    ln_coll = LineCollection(segs_aa1[n], colors = cols,linewidths=0.5, zorder=-5)
    ax11.add_collection(ln_coll)
    ax11.set_rasterization_zorder(0)

meanmodel = np.hstack((vs_mean,dep,psi2amp_mean,psi2azi_mean))
vp, vs, h, a2, psi2 = Model.get_vp_vs_h(meanmodel, vpvs_mean, mantle)
rho = vp * 0.32 + 0.77
c1 = a2 * np.cos(2*psi2)
c2 = a2 * np.sin(2*psi2)
for n, target in enumerate(targets):
    target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                              c1=c1, c2=c2)
    phveliso = target.moddata.c1
    ax11.plot(target.moddata.x,phveliso,'--',color='k')
ax11.set_xscale('log')
ax11.set_xticks([2.5,4,6,10,20,50])
ax11.set_xticklabels(["2.5","4","6","10","20","50",])
ax11.set_xlabel("$period~[s]$",labelpad=labelpad)
ax11.set_ylabel("$c_1$",labelpad=labelpad)
##########################################################################################################################
ax12 = fig.add_subplot(gs[2,0])
ax12.set_title("(c)",loc='left',fontsize=14)

ax12.errorbar(target.obsdata.x, target.obsdata.c2, yerr=target.obsdata.c2_err,
              label=label,color='blue', fmt='o', ms=2, lw=1.5, zorder=1000)    

segs_phvel = []
segs_aa1 = []
segs_aa2 = []
for n, target in enumerate(targets):
    segs_phvel.append([])
    segs_aa1.append([])
    segs_aa2.append([])
cols = []
aziamp = False
for i in misfitsU.argsort()[::-1]:
    vp, vs, h, a2, psi2 = Model.get_vp_vs_h(modelsU[i], vpvsU[i], mantle)
    rho = vp * 0.32 + 0.77
    c1 = a2 * np.cos(2*psi2)
    c2 = a2 * np.sin(2*psi2)
    for n, target in enumerate(targets):
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
    cols.append(cmap(norm(misfitsU[i]),alpha=0.9))
for n, target in enumerate(targets):
    ln_coll = LineCollection(segs_aa2[n], colors = cols,linewidths=0.5, zorder=-5)
    ax12.add_collection(ln_coll)
    ax12.set_rasterization_zorder(0)

meanmodel = np.hstack((vs_mean,dep,psi2amp_mean,psi2azi_mean))
vp, vs, h, a2, psi2 = Model.get_vp_vs_h(meanmodel, vpvs_mean, mantle)
rho = vp * 0.32 + 0.77
c1 = a2 * np.cos(2*psi2)
c2 = a2 * np.sin(2*psi2)
for n, target in enumerate(targets):
    target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                              c1=c1, c2=c2)
    phveliso = target.moddata.c2
    ax12.plot(target.moddata.x,phveliso,'--',color='k')
ax12.set_xscale('log')
ax12.set_xticks([2.5,4,6,10,20,50])
ax12.set_xticklabels(["2.5","4","6","10","20","50",])
ax12.set_xlabel("$period~[s]$",labelpad=labelpad)
ax12.set_ylabel("$c_2$",labelpad=labelpad)

##########################################################################################################################
ax2 = fig.add_subplot(gs[:,2])
ax2.set_title("(f)",loc='left',fontsize=14)
ax3 = ax2.inset_axes([1.02,0.,0.2,1.0])
#ax3 = fig.add_subplot(gs[:,3],sharey=ax2)
#ax3.set_title("(c)",loc='left',fontsize=14)
 
maxdepth = int(np.ceil(dep_int.max()))
interp = dep_int[1] - dep_int[0]
dpint = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
depbins = np.arange(0, maxdepth + 2*interp, interp)  # interp km bins
# nbin = np.arange(0, maxdepth + interp, interp)  # interp km bins

# get interfaces, #first
models2 = ModelMatrix._replace_zvnoi_h(models)
models2 = np.array([model[~np.isnan(model)] for model in models2],dtype='object')
yinterf = np.array([np.cumsum(model[int(model.size/4):-1])
                    for model in models2],dtype='object')
yinterf = np.concatenate(yinterf)

if vss_int is None:
    vss_int, psi2amps_int, psi2azis_int, deps_int = ModelMatrix.get_interpmodels(models, dpint)

vss_flatten = vss_int.flatten()
vsinterval = 0.025  # km/s, 0.025 is assumption for vs_round
# vsbins = int((vss_flatten.max() - vss_flatten.min()) / vsinterval)
vs_histmin = vs_round(vss_flatten.min())-2*vsinterval
vs_histmax = vs_round(vss_flatten.max())+3*vsinterval
vsbins = np.arange(vs_histmin, vs_histmax, vsinterval) # some buffer

# data2d, xedges, yedges = np.histogram2d(vss_flatten, deps_int.flatten(),
#                         				bins=(vsbins, depbins))
# ax2.imshow(data2d.T, extent=(xedges[0], xedges[-1],yedges[0], yedges[-1]),
#            origin='lower', aspect='auto',norm = colors.LogNorm(vmin=1,vmax=len(models)))
ax2.hist2d(vss_flatten,deps_int.flatten(),bins=[vsbins,depbins],
           cmap=plt.cm.hot_r,norm = colors.LogNorm(vmin=1,vmax=len(models)),
           rasterized=True)

# plot mean / modes
# colors = ['green', 'white']
# for c, choice in enumerate(['mean', 'mode']):
cols = ['lightgreen','cornflowerblue','black']
ls = ['-', '-', ':']
for c, choice in enumerate(['mode','mean','stdminmax']):
    vs, dep = singlemodels[choice]
    color = cols[c]
    ax2.plot(vs.T, dep, color=color, lw=2, ls=ls[c], alpha=0.9, label=choice)

vs_mode, dep_mode = singlemodels['mode']
ax2.legend(loc=3)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[:3],labels=labels[:3],loc=3)

# histogram for interfaces
data = ax3.hist(yinterf, bins=depbins, orientation='horizontal',
                    color='gray')
bins, lay_bin, _ = np.array(data,dtype='object').T
center_lay = (lay_bin[:-1] + lay_bin[1:]) / 2.

ax2.grid(visible=True)
ax2.set_ylabel('$depth~[km]$',labelpad=labelpad)
ax2.set_xlabel('$v_S~[km/s]$',labelpad=labelpad)
ax3.set_xlabel('$interface$\n$counts$',fontsize=8)
ax2.invert_yaxis()
ax3.set_xticks([])
#ax3.set_yticks([])
ax3.set_yticklabels([])
ax3.set_ylim(ax2.get_ylim())
ax3.set_xlim((0,np.max(data[0][:100])))

#####################################################################################################################

ax4 = fig.add_subplot(gs[1,1])
ax4.set_title("(d)",loc='left',fontsize=14) 
files = obj.likefiles[0] + obj.likefiles[1]
unifiles = set([f.replace('p1', 'p2') for f in files])
base = plt.cm.get_cmap(name='rainbow')
color_list = base(np.linspace(0, 1, len(unifiles)))

xmin = -obj.initparams['iter_burnin']
xmax = obj.initparams['iter_main']

files.sort()
n = 0
for i, file in enumerate(files):
    phase = int(op.basename(file).split('_p')[1][0])
    alpha = (0.4 if phase==1 else 0.7)
    ls = ('-' if phase==1 else '-')
    lw = (0.5 if phase==1 else 0.8)
    chainidx, _, ftype = obj._return_c_p_t(file)
    color = color_list[n]

    data = np.load(file)
    try:
        temperatures = np.load(file.replace(ftype,"temperatures"))
    except:
        temperatures = np.ones(len(data))
    iters = (np.linspace(xmin, 0, data.size) if phase==1 else
             np.linspace(0, xmax, data.size))
    label = 'c%d' % (chainidx)

    if not (temperatures==1.).all():
        points = np.array([iters,data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        linecolors = [color if temperatures[i]==1. else 'lightgrey' 
                      for i in range(len(data))]
        lc = LineCollection(segments, colors=linecolors,
                            linewidths=lw, alpha=alpha,zorder=-1)
        ax4.add_collection(lc)
        # ax4.plot([], [], color=color,
        #         ls=ls, lw=lw, alpha=alpha,
        #         label=label if phase==2 else '')
        ax4.set_rasterization_zorder(0)
    else:
        ax4.plot(iters, data, color=color,
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

ax4.set_xlim(xmin-5000, xmax+5000)
ax4.set_ylim(datamin*0.8, datamax*1.05)
ax4.axvline(0, color='k', ls=':', alpha=0.7)

(abs(xmin) + xmax)
center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]) / (abs(xmin) + xmax)
for i, text in enumerate(['Burn-in\n phase', 'Exploration phase']):
    ax4.text(center[i], 0.97, text,
            fontsize=8, color='k',
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax4.transAxes)

ax4.set_xlabel('$iteration$',labelpad=labelpad)
ax4.set_ylabel('$log$-$likelihood$',labelpad=labelpad)
ax4.set_xticks([-50000,0,50000,100000])
ax4.set_xticklabels(["-50k","0","50k","100k"])
#ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#####################################################################################################################

ax5 = fig.add_subplot(gs[2,1])
ax5.set_title("(e)",loc='left',fontsize=14) 
nchains=25
files = obj.misfiles[0][:nchains] + obj.misfiles[1][:nchains]
unifiles = set([f.replace('p1', 'p2') for f in files])
base = plt.cm.get_cmap(name='rainbow')
color_list = base(np.linspace(0, 1, len(unifiles)))

xmin = -obj.initparams['iter_burnin']
xmax = obj.initparams['iter_main']

files.sort()
n = 0
for i, file in enumerate(files):
    phase = int(op.basename(file).split('_p')[1][0])
    alpha = (0.4 if phase==1 else 0.7)
    ls = ('-' if phase==1 else '-')
    lw = (0.5 if phase==1 else 0.8)
    chainidx, _, ftype = obj._return_c_p_t(file)
    color = color_list[n]

    data = np.load(file)
    try:
        temperatures = np.load(file.replace(ftype,"temperatures"))
    except:
        temperatures = np.ones(len(data))
    data = data.T[-1]
    iters = (np.linspace(xmin, 0, data.size) if phase==1 else
             np.linspace(0, xmax, data.size))
    label = 'c%d' % (chainidx)

    if not (temperatures==1.).all():
        points = np.array([iters,data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        linecolors = [color if temperatures[i]==1. else 'lightgrey' 
                      for i in range(len(data))]
        lc = LineCollection(segments, colors=linecolors,
                            linewidths=lw, alpha=alpha,zorder=-1)
        ax5.add_collection(lc)
        # ax5.plot([], [], color=color,
        #         ls=ls, lw=lw, alpha=alpha,
        #         label=label if phase==2 else '')
        ax5.set_rasterization_zorder(0)
    else:
        ax5.plot(iters, data, color=color,
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

ax5.set_xlim(xmin-5000, xmax+5000)
ax5.set_ylim(datamin*0.95, datamax*1.5)
ax5.axvline(0, color='k', ls=':', alpha=0.7)

(abs(xmin) + xmax)
center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]) / (abs(xmin) + xmax)
for i, text in enumerate(['Burn-in\n phase', 'Exploration phase']):
    ax5.text(center[i], 0.97, text,
            fontsize=8, color='k',
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax5.transAxes)

ax5.set_xlabel('$iteration$',labelpad=labelpad)
ax5.set_ylabel('$joint~misfit$',labelpad=labelpad)
ax5.set_xticks([-50000,0,50000,100000])
ax5.set_xticklabels(["-50k","0","50k","100k"])
#ax5.set_yticks([0.02,0.04,0.06,0.08,0.1])
#ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))

##########################################################################################################################
ax21 = fig.add_subplot(gs[:,3])
ax21.set_title("(g)",loc='left',fontsize=14)
ax22 = ax21.inset_axes([1.05,0.,1.0,1.0])
#ax22 = fig.add_subplot(gs[:,4],sharey=ax2)
ax22.set_title("(h)",loc='left',fontsize=14)

# Variant 2: plot histograms
cols = [(1, 1, 1), (0, 0, 1)] # first color is white, last is red
cmap = colors.LinearSegmentedColormap.from_list("Custom", cols, N=50)
h1 = ax21.hist2d(singlemodels['psi2amp'].flatten()*100,
                 np.tile(dep,len(singlemodels['psi2amp'])),
                 bins=[np.linspace(0,np.max(psi2amp)*100,31),dep],cmap=plt.cm.hot_r,density=True)
h1[3].set_clim(0,np.max(h1[0])/2)

misfits_azi = misfits[uidx_azi]
#ax2.scatter(np.hstack(psi2azi)/np.pi*180,np.tile(dep,len(psi2azi)),c=np.tile(misfits_azi,len(dep)),s=1,cmap=cm.jet_r,zorder=-5)
valid = singlemodels['psi2amp'].flatten()>0.
h2 = ax22.hist2d(np.hstack(singlemodels['psi2azi'])[valid]/np.pi*180,
                 np.tile(dep,len(singlemodels['psi2azi']))[valid],
                 bins=[np.linspace(0,180,31),dep],cmap=plt.cm.hot_r,density=True)
h2[3].set_clim(0,np.max(h2[0])/2)
ax2.set_rasterization_zorder(0)

ax21.plot(psi2amp_mean.T*100, dep, color='cornflowerblue', label='mean', ls='-', lw=2)
ax21.plot((psi2amp_mean+psi2amp_std).T*100, dep, 'k', ls=':', label='std', lw=2)
lowstd = (psi2amp_mean-psi2amp_std).T*100
lowstd[lowstd<0.] = 0.
ax21.plot(lowstd, dep, 'k', ls=':', lw=2)
        
p2am = np.split(psi2azi_mean,np.where(np.abs(np.diff(psi2azi_mean))>np.pi/2.)[0]+1)
dp = np.split(dep,np.where(np.abs(np.diff(psi2azi_mean))>np.pi/2.)[0]+1)
for i in range(len(p2am)):
    ax22.plot(p2am[i]/np.pi*180, dp[i], color='cornflowerblue', lw=2)
stdlevel = np.zeros(len(psi2azi_std))
stdlevel[psi2azi_std>=0.52] = 1
stdlevel[psi2azi_std>=0.8] = 2
#stdmax = psi2azi_std>=0.8 # maximum possible std: np.std(np.random.uniform(-np.pi/2.,np.pi/2.,10000)) ~ 0.9
cdict = {True: "red",False: "black"}
highstd = psi2azi_mean+psi2azi_std
highstd[highstd>np.pi] -= np.pi
split = np.where((np.abs(np.diff(highstd))>np.pi/2.)+(np.diff(stdlevel)>0))[0]+1
p2am = np.split(highstd,split)
dp = np.split(dep,split)
smax = np.split(stdlevel,split)
for i in range(len(p2am)):
    for j,col in enumerate(['black','gray']):#,'lightgray']):
        ax22.plot(p2am[i][smax[i]==j]/np.pi*180, dp[i][smax[i]==j], color=col, ls='--',lw=1.5)
    #ax22.plot(p2am[i][~smax[i]]/np.pi*180, dp[i][~smax[i]], color='black', ls='--', lw=1.5) # std
    #ax22.plot(p2am[i][smax[i]]/np.pi*180, dp[i][smax[i]], color='red', ls='--', lw=1.5) # std (maximized)
lowstd = psi2azi_mean-psi2azi_std
lowstd[lowstd<0.] += np.pi
split = np.where((np.abs(np.diff(lowstd))>np.pi/2.)+(np.diff(stdlevel)>0))[0]+1
p2am = np.split(lowstd,split)
dp = np.split(dep,split)
smax = np.split(stdlevel,split)
for i in range(len(p2am)):
    for j,col in enumerate(['black','gray']):#,'lightgray']):
        ax22.plot(p2am[i][smax[i]==j]/np.pi*180, dp[i][smax[i]==j], color=col, ls='--',lw=1.5)
    #ax22.plot(p2am[i][~smax[i]]/np.pi*180, dp[i][~smax[i]], color='black', ls='--', lw=1.5)
    #ax22.plot(p2am[i][smax[i]]/np.pi*180, dp[i][smax[i]], color='red', ls='--', lw=1.5)
ax22.plot([],[],color='black',ls='--',lw=1.5,label='$std<30^{\circ}$')
ax22.plot([],[],color='gray',ls='--',lw=1.5,label='$30^{\circ}\leq std\leq 45^{\circ}$')
#ax22.plot([],[],color='lightgray',ls='--',lw=1.5,label='$std>0.8$')
ax22.legend(loc='lower left')#,bbox_to_anchor=(-0.2,0))

ax21.set_xlabel('$a_2~[\%]$',labelpad=labelpad)
ax22.set_xlabel('$\Psi_2~[math.~deg.]$',labelpad=labelpad)
ax22.set_xticks([0,90,180])

ax21.set_yticklabels([])
ax22.set_yticklabels([])
ax21.grid(visible=True)
ax22.grid(visible=True)

ax21.set_ylim([120,0])
ax22.set_ylim(ax21.get_ylim())


# #####################################################################################################################
# ax6 = fig.add_subplot(gs[0,1])
# ax6.set_title("(d)",loc='left',fontsize=14) 
# count, bins, _ = ax6.hist(vpvs, bins=30, color='darkblue', alpha=0.7,
#                          edgecolor='white', linewidth=0.4)
# text = 'mean: %s' % '%.2f' % vpvs_mean
# ax6.text(0.9, 0.97, text,
#         fontsize=9, color='k',
#         horizontalalignment='right',
#         verticalalignment='top',
#         transform=ax6.transAxes)
# ax6.axvline(vpvs_mean, color='k', ls=':', lw=1)
# # xticks = np.array(ax.get_xticks())
# # ax.set_xticklabels(xticks, fontsize=8)
# ax6.set_yticks([])
# ax6.spines['top'].set_visible(False)
# ax6.spines['right'].set_visible(False)
# ax6.set_xlabel("$v_P/v_S~ratio$")

# #####################################################################################################################
# ax7 = fig.add_subplot(gs[1,1])
# ax7.set_title("(e)",loc='left',fontsize=14)
# validmods = np.array([model[~np.isnan(model)] for model in models],dtype='object')
# layers = np.array([(model.size/4 - 1) for model in validmods],dtype='object')
# bins = np.arange(np.min(layers), np.max(layers)+2)-0.5
# count, bins, _ = ax7.hist(layers, bins=bins, color='darkblue', alpha=0.7,
#                          edgecolor='white', linewidth=0.4)
# # text = 'mean: %s' % '%.2f' % vpvs_mean
# # ax6.text(0.8, 0.97, text,
# #         fontsize=9, color='k',
# #         horizontalalignment='right',
# #         verticalalignment='top',
# #         transform=ax6.transAxes)
# # ax6.axvline(vpvs_mean, color='k', ls=':', lw=1)
# # xticks = np.array(ax.get_xticks())
# # ax.set_xticklabels(xticks, fontsize=8)
# ax7.set_yticks([])
# ax7.spines['top'].set_visible(False)
# ax7.spines['right'].set_visible(False)
# ax7.set_xlabel("$no~layers$")

# #####################################################################################################################
# ax8 = fig.add_subplot(gs[2,1])
# ax8.set_title("(f)",loc='left',fontsize=14) 
# bins = np.arange(0.02,0.052,.001)
# count, bins, _ = ax8.hist(noise[:,1], bins=bins, color='darkblue', alpha=0.7,
#                          edgecolor='white', linewidth=0.4)
# text = 'mean: %s' % '%.3f' % raystd_mean
# ax8.text(0.9, 0.97, text,
#         fontsize=9, color='k',
#         horizontalalignment='right',
#         verticalalignment='top',
#         transform=ax8.transAxes)
# ax8.axvline(raystd_mean, color='k', ls=':', lw=1)
# ax8.set_yticks([])
# ax8.spines['top'].set_visible(False)
# ax8.spines['right'].set_visible(False)
# ax8.set_xlabel("$\sigma_R~[km/s]$")
# ax8.set_xlim(0.015,0.045)
# ##############################################################################
# ax9 = fig.add_subplot(gs[3,1])
# ax9.set_title("(g)",loc='left',fontsize=14) 
# count, bins, _ = ax9.hist(noise[:,5], bins=bins, color='darkblue', alpha=0.7,
#                          edgecolor='white', linewidth=0.4)
# text = 'mean: %s' % '%.3f' % raystd_mean
# ax9.text(0.9, 0.97, text,
#         fontsize=9, color='k',
#         horizontalalignment='right',
#         verticalalignment='top',
#         transform=ax9.transAxes)
# ax9.axvline(raystd_mean, color='k', ls=':', lw=1)
# ax9.set_yticks([])
# ax9.spines['top'].set_visible(False)
# ax9.spines['right'].set_visible(False)
# ax9.set_xlabel("$\sigma_L~[km/s]$")
# ax9.set_xlim(0.015,0.045)

# for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

plt.savefig("methodplot.jpg",bbox_inches='tight',dpi=300)
plt.savefig("methodplot.pdf",bbox_inches='tight',dpi=100)
plt.show()


