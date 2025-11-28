from setuptools import setup, Extension
import numpy as np

# Define the C extension
ext_module = Extension(
    'fast_bocpd._core',
    sources=[
        'fast_bocpd/_c/bocpd_core.c',
        'fast_bocpd/_c/gaussian_nig.c',
        'fast_bocpd/_c/hazard.c',
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=['-std=c99', '-O3', '-Wall', '-Wextra'],
    extra_link_args=['-lm'],
)

setup(
    name='fast-bocpd',
    version='0.1.0',
    description='Fast Bayesian Online Changepoint Detection with C backend',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/Fast_BOCPD',
    packages=['fast_bocpd'],
    ext_modules=[ext_module],
    install_requires=[
        'numpy>=1.20.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
