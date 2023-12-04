# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os, sys
from mpi4py import MPI
# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
plt.ioff()
import shutil, glob
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from BayHunter import PlotFromStorage
from BayHunter import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer
from BayHunter import ModelMatrix, Model
from BayHunter import SynthObs
import logging
#own plugin (a bit faster than pysurf, but more difficult to install)
# from BayHunter.dccurve_modsw import SurfDisp_dccurve
# if not 'dccurve' in sys.modules:
#     import BayHunter.dccurve_ext as dccurve
#     dccurve.init_dccurve(0)
# else:
#     import BayHunter.dccurve_ext as dccurve

# some convenience functions
def gaussdist(x, mu, sig, sig_trunc=None):
    if sig==0.:
        sig = 1e-10
    gauss = (1. / (np.sqrt(2 * np.pi) * sig) * np.exp(
                -0.5 * np.square(x - mu) / np.square(sig)))
    if sig_trunc is not None:
        bounds = (mu-sig_trunc*sig,mu+sig_trunc*sig)
        gauss[x < bounds[0]] = 0
        gauss[x > bounds[1]] = 0
    return gauss/np.sum(gauss)
        
def gaussian_filt(U,sigma,truncate):
    """ Works like gaussian_filter but can handle NaNs in the array """
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma,truncate=truncate)
    
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma,truncate=truncate)
    
    return VV/WW

def running_mean(x, N):
    """ running mean / moving average that can handle NaNs """
    if N>len(x):
        print("Warning: Length of the array is shorter than the number of samples to apply the running mean. Setting N=%d." %len(x))
        N=len(x)
    if N<=1 or np.isnan(x).all():
        return x
    if N%2 == 0:
        N+=1
    didx = int((N-1)/2)
    runmean = np.ones(len(x))*np.nan
    fu = interp1d(np.arange(len(x))[~np.isnan(x)],x[~np.isnan(x)],
                  bounds_error=False,fill_value='extrapolate')
    xint = fu(np.arange(len(x)))
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        ddidx = didx
        if i-ddidx<0:
            ddidx = i
        if i+ddidx>=len(x):
            ddidx = len(x)-i-1
        idx = np.linspace(i-ddidx,i+ddidx,N,dtype=int)
        if np.min(idx)<0:
            raise Exception("1")
        if np.max(idx)>=len(x):
            raise Exception("2")
        #idx[idx<0] = 0
        #idx[idx>=len(x)] = len(x)-1
        runmean[i] = np.mean(xint[idx])
    return runmean

#%% RUN BAYHUNTER MODEL SEARCH
#
# console printout formatting
#
formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger()

# # threads per process, should be 1 if using mpi
nthreads = 4

# Initialize MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size() 
#
# ------------------------------------------------------------ set paths
#
# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)
finalmodels_path = os.path.join(initparams['savepath'],"finalmodels")

try:
    array_index = int(sys.argv[1])
    array_size = int(sys.argv[2])
except:
    array_index = None
    
if array_index is None:
    if mpi_rank==0:
        if not os.path.exists(finalmodels_path):
            os.makedirs(finalmodels_path)
            os.makedirs(finalmodels_path.replace("finalmodels","rayleigh_mean"))
            os.makedirs(finalmodels_path.replace("finalmodels","love_mean"))
    initparams.update({'savepath':"results_test_%d" %mpi_rank})
else:
    if array_index==0:
        if not os.path.exists(finalmodels_path):
            os.makedirs(finalmodels_path)
            os.makedirs(finalmodels_path.replace("finalmodels","rayleigh_mean"))
            os.makedirs(finalmodels_path.replace("finalmodels","love_mean"))
    initparams.update({'savepath':"results_%d" %array_index})
path = initparams['savepath']

# ------------------------------------------------------- distribute workload 
files = glob.glob("rayleigh_profiles/*.txt")

# set up a worklist   
worklist = []
for i in np.arange(len(files),dtype=int):
    lon = files[i].split("_")[-2]
    lat = files[i].split("_")[-1].split(".txt")[0]
    finalmod_filename = os.path.join(finalmodels_path,"finalmod_%s_%s.txt" %(lon,lat))
    # skip if already processed
    #if os.path.isfile(finalmod_filename):
    #    continue
    worklist.append(i)    
   
# job distribution
if array_index is not None:
    worklist = worklist[array_index::array_size]
    print("job number %d working on profiles" %array_index,worklist)
    np.random.shuffle(worklist)
else: # distribute workload on all availabe MPI processes
    if mpi_rank==0:
        np.random.shuffle(worklist)
    else:
        worklist = []
    worklist = mpi_comm.bcast(worklist,root=0) 
    worklist = worklist[mpi_rank::mpi_size]

counter = 0
test = []
stddata = []
for i in worklist:
    
    counter += 1
    #print("MPI rank",mpi_rank,"array index",array_index,"working on",counter,"/",len(worklist))
    
    # # LOAD INPUT DATA # # 
    lon = files[i].split("_")[-2]
    lat = files[i].split("_")[-1].split(".txt")[0]

    result_path = os.path.join(path,"data_%s_%s" %(lat,lon))
    finalmod_filename = os.path.join(finalmodels_path,"finalmod_%s_%s.txt" %(lon,lat))
    
    #if not "47.500_13.375" in result_path:
    #    continue
            
    data = np.loadtxt(files[i])
    data = data[data[:,0].argsort()]
    xray = data[:,0]
    yray = data[:,1]
    valid = ~np.isnan(yray)*(xray>=2.5)*(xray<=50.)*(data[:,2]<0.3)
    veljumps = np.where(np.diff(yray)<-0.2)[0]
    if len(veljumps)>0:
        valid[:veljumps[-1]+1] = False
    if np.sum(valid)<5:
        continue
    xray = xray[valid]
    yray = yray[valid]
    # the _err values give the standard deviations
    yray_err = data[valid,2]
    yray_c1 = data[valid,3]
    yray_c1err = data[valid,4]
    yray_c2 = data[valid,5]
    yray_c2err = data[valid,6]
    
    data_lov = np.loadtxt(files[i].replace("rayleigh","love"))
    data_lov = data_lov[data_lov[:,0].argsort()]
    xlov = data_lov[:,0]
    ylov = data_lov[:,1]
    valid = ~np.isnan(ylov)*(xlov>=2.5)*(xlov<=50.)*(data_lov[:,2]<0.3)
    veljumps = np.where(np.diff(ylov)<-0.2)[0]
    if len(veljumps)>0:
        valid[:veljumps[-1]+1] = False
    if np.sum(valid)<5:
        continue
    xlov = xlov[valid]
    ylov= ylov[valid]
    
    # # SET TARGETS # # 
    
    # Only pass x and y observed data to the Targets object which is matching
    # the data type. You can chose for SWD any combination of Rayleigh, Love, group
    # and phase velocity. Default is the fundamendal mode, but this can be updated.
    # For RF chose P or S. You can also use user defined targets or replace the
    # forward modeling plugin wih your own module.
    
    # Here, we give the variance (std squared) as error. The _err values are
    # only used for plotting and, if defined, to relatively scale the standard
    # deviations at different periods.
    target1 = Targets.RayleighDispersionPhase(xray, yray, yerr=yray_err**2,
                                              c1=yray_c1, c1err=yray_c1err**2,
                                              c2=yray_c2, c2err=yray_c2err**2
                                              )    
    #target1.update_plugin(SurfDisp_dccurve(target1.obsdata.x,target1.ref))

    # target2 (Love) has no anisotropy
    target2 = Targets.LoveDispersionPhase(xlov, ylov)
    #target2.update_plugin(SurfDisp_dccurve(target2.obsdata.x,target2.ref))
    
    # join both targets
    targets = [target1,target2]
    
    # Join the targets. targets must be a list instance with all targets
    # you want to use for MCMC Bayesian inversion.
    targets = Targets.JointTarget(targets=targets)

    # # MODIFY PRIORS AND PARAMS (otherwise, values from config.ini are chosen) # # 
    #
    #  ---------------------------------------------------  Quick parameter update
    #
    # "priors" and "initparams" from config.ini are python dictionaries. You could
    # also simply define the dictionaries directly in the script, if you don't want
    # to use a config.ini file. Or update the dictionaries as follows, e.g. if you
    # have station specific values, etc.
    # See docs/bayhunter.pdf for explanation of parameters
    
    # priors.update({'mohoest': (38, 4),  # optional, moho estimate (mean, std)
    #                'rfnoise_corr': 0.98,
    #                'swdnoise_corr': 0.
    #                # 'rfnoise_sigma': np.std(yrf_err),  # fixed to true value
    #                # 'swdnoise_sigma': np.std(ysw_err),  # fixed to true value
    #                })
    
    if True: # optional: change some model search parameters
        initparams.update({'savepath': 'rayleigh_love_joint',
                            })
    # update noise priors
    # if only a single value is given, the standard deviation is fixed to this value
    # otherwise do a hierarchical search, where the std is allowed within the prior range
    if False: 
        # fix standard deviations to the (smallest) value from the input file
        priors.update({'swdnoise_sigma': np.min(yray_err), #(np.min(yray_err),0.25),
                       'swdnoise_sigma_c1': np.min(yray_c1err), #(np.min(yray_c1err),0.025),
                       'swdnoise_sigma_c2': np.min(yray_c2err)}) #(np.min(yray_c2err),0.025),})
        
        

    #
    #  ---------------------------------------------------  MCMC BAY INVERSION
    #
    # Save configfile for baywatch. refmodel must not be defined.
    utils.save_baywatch_config(targets, path='.', priors=priors,
                               initparams=initparams)
    optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                               random_seed=None)
    # default for the number of threads is the amount of cpus == one chain per cpu.
    # if baywatch is True, inversion data is continuously send out (dtsend)
    # to be received by BayWatch (see below).
    optimizer.mp_inversion(nthreads=nthreads, baywatch=False, dtsend=1)

    #
    # #  ----------------------------------------- Model resaving and plotting
    cfile = '%s_config.pkl' % initparams['station']
    configfile = op.join(path, 'data', cfile)
    obj = PlotFromStorage(configfile)    

    # The final distributions will be saved with save_final_distribution.
    # Beforehand, outlier chains will be detected and excluded.
    # Outlier chains are defined as chains with a likelihood deviation
    # of dev * 100 % from the median posterior likelihood of the best chain.
    obj.save_final_distribution(maxmodels=100000, dev=10)

    depint = 1
    models, = obj._get_posterior_data(["models"],final=True)
    vpvs, = obj._get_posterior_data(['vpvs'], final=True)
    vpvs_mean = np.mean(vpvs)
    noise, = obj._get_posterior_data(['noise'], final=True)
    raystd_mean = np.mean(noise[:,1])
    try:
        lovstd_mean = np.mean(noise[:,3])
    except:
        print("no love waves")
        lovstd_mean = 0.
    dep_int = np.arange(obj.priors['z'][0],obj.priors['z'][1], depint)
    model_elements = ['mean', 'median', 'stdminmax']
    singlemodels = ModelMatrix.get_singlemodels(models, dep_int)
    vs_mean, dep = singlemodels['mean']
    vs_median, dep = singlemodels['median']
    vs_mode, depi = singlemodels['mode']
    vs_mode = np.interp(dep,depi,vs_mode,)
    vs_std, dep = singlemodels['stdminmax']
    psi2amp_mean, dep = singlemodels['psi2amp_mean']
    psi2azi_mean, dep = singlemodels['psi2azi_mean']
    psi2amp_std, dep = singlemodels['psi2amp_std']
    psi2azi_std, dep = singlemodels['psi2azi_std']
    if psi2amp_mean is None:
        psi2amp_mean = psi2azi_mean = psi2amp_std = psi2azi_std = np.zeros_like(vs_mean)
        
    # get mean model
    ns = 1 # additional thinning, take every ns-th model
    #if len(vs_mean)>100:
    #    ns = int(np.ceil(len(vs_mean)/100))
    meanmodel = np.hstack((vs_mean[::ns],dep[::ns],psi2amp_mean[::ns],psi2azi_mean[::ns]))
    vp, vs, h, psi2amp, psi2azi = Model.get_vp_vs_h(meanmodel, vpvs_mean, obj.mantle)
    rho = vp * 0.32 + 0.77
    c1 = psi2amp * np.cos(2*psi2azi)
    c2 = psi2amp * np.sin(2*psi2azi)
    for n, target in enumerate(obj.targets):
        target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, 
                                  c1=c1, c2=c2)
        phveliso = target.moddata.y
        if target.moddata.azimuthal_anisotropic:
            anisoamp = target.moddata.aa_amp
            anisoazi = target.moddata.aa_ang
        else:
            anisoamp = None
        #phveliso,anisoamp,anisoazi = target.moddata.get_anisotropy()
        if target.ref.startswith("r"):
            writepath = finalmodels_path.replace("finalmodels","rayleigh_mean")
        else:
            writepath = finalmodels_path.replace("finalmodels","love_mean")
        if anisoamp is None:
            np.savetxt(os.path.join(writepath,os.path.basename(files[i])),
                       np.column_stack((target.obsdata.x,phveliso)),
                       fmt="%5.1f %.2f")       
        else:
            np.savetxt(os.path.join(writepath,os.path.basename(files[i])),
                       np.column_stack((target.obsdata.x,phveliso,anisoamp,anisoazi)),
                       fmt="%5.1f %.2f %.1f %.2f")

    # get histogram of interface depths
    models2 = ModelMatrix._replace_zvnoi_h(models)
    models2 = np.array([model[~np.isnan(model)] for model in models2],dtype='object')
    yinterf = np.array([np.cumsum(model[int(model.size/4):-1])
                        for model in models2],dtype='object')
    yinterf = np.concatenate(yinterf)
    maxdepth = int(np.ceil(dep_int.max()))
    interp = dep_int[1] - dep_int[0]
    dep_int = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
    depbins = np.arange(0, maxdepth + 2*interp, interp)
    stats = binned_statistic(yinterf,[],statistic='count',bins=depbins)
    header = "vpvs_crust=%.2f ray_std=%.3f lov_std=%.3f\n" %(vpvs_mean,raystd_mean,lovstd_mean)
    header += "depth vs_mean vs_median vs_mode vs_std_low vs_std_high interface_count anisoamp_mean anisoamp_std anisoazi_mean[radians, mathematical angles] anisoazi_std"
    finalmod = np.column_stack((dep,vs_mean,vs_median,vs_mode,vs_std[0],
                                vs_std[1],stats.statistic,psi2amp_mean*100,psi2amp_std*100,psi2azi_mean,psi2azi_std))
    #np.save(finalmod_filename,finalmod)
    np.savetxt(finalmod_filename,finalmod,header=header,
               fmt="%5.1f %5.3f %5.3f %5.3f %5.3f %5.3f %4d %.1f %.1f %.2f %.2f")
    
    # Save a selection of important plots
    obj.save_plots(nchains=15,dpi=100)#initparams['nchains'])
    obj.merge_pdfs()
    summaryplots = glob.glob(os.path.join(path,"c_summary.pdf"))[0]
    shutil.copyfile(summaryplots,"/".join(finalmodels_path.split("/")[:-1])+"/plots_%s_%s.pdf" %(lon,lat))
    if False: # to save disk space, you may want to delete old files
        shutil.rmtree(os.path.join(path,"data"))
    


# #  ---------------------------------------------- WATCH YOUR INVERSION
# if you want to use BayWatch, simply type "baywatch ." in the terminal in the
# folder you saved your baywatch configfile or type the full path instead
# of ".". Type "baywatch --help" for further options.

# if you give your public address as option (default is local address of PC),
# you can also use BayWatch via VPN from 'outside'.
# address = '139.?.?.?'  # here your complete address !!!


if False:
    avgmod = np.load("dinver_comparison/avgmodels/bestavgmodel_131.962_-10.317.npy")
    fig = obj.plot_posterior_models1d()
    ax = fig.axes[0]
    ax.plot(avgmod[:,2],avgmod[:,0],'r',label='from dinver')
    ax.legend(loc='lower left')
    fig.savefig("comparisonplot.pdf",dpi=200,bbox_inches='tight')

    #%%
