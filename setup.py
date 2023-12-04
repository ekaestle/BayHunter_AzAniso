#!/usr/bin/env python
try:
    from numpy.distutils.core import Extension as NumpyExtension
    from numpy.distutils.core import setup

    from distutils.extension import Extension
    from Cython.Build import cythonize

    import numpy

    from shutil import copyfile

except ImportError:
    raise ImportError('Numpy needs to be installed or updated.')


extensions = [
    NumpyExtension(
        name='BayHunter.surfdisp96_ext',
        sources=['src/extensions/surf96aa.f'],
        extra_f77_compile_args='-O3 -ffixed-line-length-none -fbounds-check -m64'.split(),  # noqa
        f2py_options=['only:', 'depthkernel', 'surfdisp96', ':'],
        language='f77'),
    ]

# not working...
#extensions.extend([
#    NumpyExtension(
#        name='BayHunter.dccurve_ext',
#        sources=['src/extensions/dccurve.f'],
#        extra_f77_compile_args=['-Lsrc/extensions/ -lQGpCoreWave'],
#        f2py_options=[],
#        language='f77'),
#    ])

extensions.extend(cythonize(
    Extension("BayHunter.rfmini", [
        "src/extensions/rfmini/rfmini.pyx",
        "src/extensions/rfmini/greens.cpp",
        "src/extensions/rfmini/model.cpp",
        "src/extensions/rfmini/pd.cpp",
        "src/extensions/rfmini/synrf.cpp",
        "src/extensions/rfmini/wrap.cpp",
        "src/extensions/rfmini/fork.cpp"],
        include_dirs=[numpy.get_include()])))


setup(
    name="BayHunter",
    version="2.1",
    author="Jennifer Dreiling",
    author_email="jennifer.dreiling@gfz-potsdam.de",
    description=("Transdimensional Bayesian Inversion of RF and/or SWD."),
    install_requires=[],
    url="https://github.com/jenndrei/BayHunter",
    packages=['BayHunter'],
    package_dir={
        'BayHunter': 'src'},

    scripts=['src/scripts/baywatch'],

    package_data={
        'BayHunter': ['defaults/*'], },

    ext_modules=extensions
)

try:
    print("copying dccurve lib.")
    copyfile("src/extensions/dccurve_ext.cpython-39-x86_64-linux-gnu.so","/home/emanuel/BIN/anaconda3/envs/py3/lib/python3.9/site-packages/BayHunter-2.1-py3.9-linux-x86_64.egg/BayHunter/dccurve_ext.cpython-39-x86_64-linux-gnu.so")
except:
    pass
