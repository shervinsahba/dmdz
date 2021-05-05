import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

import dmdz  # used to populate some meta-data from __init__.py

# Package meta-data.
NAME = dmdz.__title__
VERSION = dmdz.__version__
LICENSE = dmdz.__license__
AUTHOR = dmdz.__author__
EMAIL = dmdz.__mail__
URL = 'https://github.com/shervinsahba/dmdz'
DESCRIPTION = 'Dynamic Mode Decomposition Toolkit'
KEYWORDS = 'dynamic-mode-decomposition dmd fbdmd optdmd spdmd'
REQUIRED_PYTHON = '>=3.8.0'
REQUIREMENTS = ['numpy', 'scipy', 'matplotlib', 'seaborn', 'svgutils', 'ipython']
EXTRAS = {}
CLASSIFIERS = [  # see https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
]


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


class UploadCommand(Command):
    """Support setup.py upload."""
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=[NAME],
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    license=LICENSE,
    python_requires=REQUIRED_PYTHON,
    install_requires=REQUIREMENTS,
    extra_requires=EXTRAS,
    zip_safe=False,
    cmdclass={  # $ setup.py publish support.
        'upload': UploadCommand,
    },
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
)
