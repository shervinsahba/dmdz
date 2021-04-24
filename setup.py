from setuptools import setup, Command
import os
import sys
import pydmdz

# Package meta-data.
NAME = pydmdz.__title__
DESCRIPTION = 'Dynamic Mode Decomposition Toolkit'
URL = 'https://github.com/shervinsahba/pydmdz'
MAIL = pydmdz.__mail__
AUTHOR = pydmdz.__author__
VERSION = pydmdz.__version__
KEYWORDS='dynamic-mode-decomposition dmd mrdmd fbdmd cdmd'

REQUIRED = [
    'numpy', 'scipy', 'matplotlib',
]

EXTRAS = {
    'docs': ['Sphinx==1.4', 'sphinx_rtd_theme'],
}

LDESCRIPTION = (
    "TBD"
)

here = os.path.abspath(os.path.dirname(__file__))
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
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
	classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
	],
	keywords=KEYWORDS,
	url=URL,
	license='MIT',
	packages=[NAME],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    test_suite='nose.collector',
	tests_require=['nose'],
	include_package_data=True,
	zip_safe=False,

    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },)
