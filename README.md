# BayHunter v2.1

BayHunter is an open source Python tool to perform an McMC transdimensional Bayesian inversion of surface wave dispersion and/or receiver functions. The algorithm follows a data-driven strategy and solves for the velocity-depth structure, the number of layers, Vp/Vs ratio and noise parameters, i.e., data noise correlation and amplitude. The forward modeling codes are provided within the package, but are easily replaceable with own codes. It is also possible to add (completely different) data sets.

The BayWatch module can be used to live-stream the inversion while it is running: this makes it easy to see how each chain is exploring the parameter space, how the data fits and models change and in which direction the inversion progresses.

**This is an updated version of the original BayHunter program to include also azimuthal anisotropy ($\Psi_2$ component). The azimuthal anisotropy option is only available for Rayleigh surface-wave dispersion. Feel free to contact E. Kästle in case of questions.**

### Differences to original BayHunter

Some parts of the program are changed, so be careful when switching from the original BayHunter to this version (the starting point for this version is the 2021 version of BayHunter)!
* the config.ini file has more entries and some have slightly different meanings (see below)
* includes 2-Psi azimuthal anisotropy for Rayleigh waves
* has a parallel tempering option
* by a combination of mpi4py and multiprocessing, has more flexible options for parallel calculations
* fixed an error in the acceptance rate optimization

### Running options

There three basic two options to run the program.
(1) Sequential run
Set nthreads=1 at the beginning of the bayhunter_....py script
Run using __python bayhunter_depthinversion...py__

(2) Parallel execution using multiprocessing
This means that one profile is inverted at a time, but for each profile, the chains run in parallel.
Set nthrads to a number greater 1 at the beginning of the bayhunter script
Run using __python bayhunter_depthinversion...py__

(3) Parallel execution using MPI
This means that several profiles are inverted in parallel, but for each profile, just a single process works on all chains.
Run using __mpirun -np no_cores python bayhunter_...py__ (replace no_cores with the desired number of parallel threads).

It's also possible to combine (2) and (3), but I don't think that would make sense, since the cores will be at full load in any case.


### config.ini file

Most parts are the same (see example file in folder). The differences are (with explanations in brackets):

[modelpriors]
* triangular_zprop = False (if True, higher probability of creating layers at shallow depth, maintains birth/death balance)
* mohoest = 40,30 (Moho depth, depth variation, can also be None. If set, then a velocity jump from smaller 4.1 to greater 4.2 at a depth of Moho depth +- depth variation is required).
* swdnoise_sigma_c1 = 1e-5, 0.02 (minimum and maximum sigma parameter for the anisotropic c1 component, fixed if only 1 value is given) 
* swdnoise_sigma_c2 = 1e-5, 0.02 (minimum and maximum sigma parameter for the anisotropic c2 component, fixed if only 1 value is given)

[initparams]
* propdist =  0.3,     3,   0.5, 0.005,  0.01 (vs,z_move,vs_birth/death, noise, vpvs)
* propfixed =   0,     0,     1,     0,     0 (whether the propdist is fixed or not)
* acceptance = 40, 48 (it is now ensured that these acceptance rates are kept for same-dimension steps, also for later iterations)
* relative_thickmin = True (if 'True' then thickmin is thickmin*depth)
* lvz = 0.5 (absolute allowed dv in km/s, not relative)
* hvz = 0.8 (absolute allowed dv in km/s, not relative)
* parallel_tempering = False (apply parallel tempering)
* t1chains = 20 (number of chains running at temperature T=1, i.e. untempered)
* maxtemp = 2.0 (chains not on T=1 follow a logspaced temperature profile from 1 to maxtemp)
* azimuthal_anisotropy = True (include azimuthal anisotropy for Rayleigh phasevels)


### Examples

There are two example files in the folder __example_kaestle_tilmann_2023__.
```bash
python bayhunter_depthinversion_rayleigh_anisotropic.py
python methodplot.py
```
Use this example to run the anisotropic depth inversion at location lon/lat=13.500/46.500 to reproduce figure 7 of the article referenced below (Kästle and Tilmann, 2023).

The other example file performs a joint inversion of Rayleigh (anisotropic) and Love (isotropic) with slightly different search parameters.


## Readme below unchanged from original BayHunter

### Citation

> Dreiling, Jennifer; Tilmann, Frederik (2019): BayHunter - McMC transdimensional Bayesian inversion of receiver functions and surface wave dispersion. GFZ Data Services. [http://doi.org/10.5880/GFZ.2.4.2019.001](http://doi.org/10.5880/GFZ.2.4.2019.001)

For the anisotropic version please also cite
> Kästle and Tilmann (2023), Anisotropic reversible-jump MCMC shear-velocity tomography of the eastern Alpine crust, submitted


### Application examples

> Dreiling et al. (2020): Crustal structure of Sri Lanka derived from joint inversion of surface wave dispersion and receiver functions using a Bayesian approach. Journal of Geophysical Research: Solid Earth. [https://doi.org/10.1029/2019JB018688](https://doi.org/10.1029/2019JB018688).

> Green et al. (2020): Magmatic and sedimentary structure beneath the Klyuchevskoy Volcanic Group, Kamchatka, from ambient noise tomography. Journal of Geophysical Research: Solid Earth. [https://doi.org/10.1029/2019JB018900](https://doi.org/10.1029/2019JB018900).

> Mauerberger et al. (n.a.): The multifaceted Scandinavian lithosphere imaged by surface waves and ambient noise. In preparation.

For the anisotropic inversion please also see


### Comments and Feedback

BayHunter is ready to use. It is quick and efficient and I am happy with the performance. Still,
there are always things that can be improved to make it even faster and more efficient, and user
friendlier. 

Although we tested the software with a variety of synthetic and real data, each data set is
still unique and shows own characteristics. If you observe any unforeseen behavior, please share
it with me to wipe out possible problems we haven’t considered.

I am happy to share my experience with you and also if you share your thoughts with me. I am looking forward to your feedback. 


### Who am I?

I am Jennifer Dreiling. I finished my PhD studies at the German Research Center for Geosciences (GFZ) in Potsdam, Germany. I created BayHunter in the frame of my PhD program. [Contact me](https://www.gfz-potsdam.de/en/staff/jennifer-dreiling/).



## Quick start

### Requirements
* matplotlib
* numpy
* PyPDF2
* configobj
* zmq
* Cython

### Installation (compatible with Python 2 and 3)*

*Although BayHunter is currently compatible with Python 2 and 3, we recommend you to upgrade to Python 3, as the official support for Python 2 has stopped in January 2020.

```sh
git clone https://github.com/jenndrei/BayHunter.git
cd BayHunter
sudo python setup.py install
```

### Documentation and Tutorial

The documentation to BayHunter offers background information on the inversion algorithm, the parameters and usage of BayHunter and BayWatch (tutorial). See the [documentation here](https://jenndrei.github.io/BayHunter/) or download the [PDF](https://github.com/jenndrei/BayHunter/blob/master/documentation/BayHunter_v2.1_documentation.pdf). Also check out the [FAQs](https://jenndrei.github.io/BayHunter/FAQs).

An example inversion can be found in the **tutorial folder**.
The file to be run, `tutorialhunt.py`, is spiked with comments.
You can also create your own synthetic data set with `create_testdata.py`.


### Resources

* Algorithm: based on the work of [Bodin et al., 2012](https://doi.org/10.1029/2011JB008560).
* SWD forward modeling: SURF96 from Computer Programs in Seismology ([CPS](http://www.eas.slu.edu/eqc/eqccps.html)). Python wrapper using [pysurf96](https://github.com/miili/pysurf96) and [SurfTomo](https://github.com/caiweicaiwei/SurfTomo).
* RF forward modeling: **rfmini** from [Joachim Saul, GFZ](https://www.gfz-potsdam.de/en/staff/joachim-saul/).
