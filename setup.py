from setuptools import setup, find_packages

setup(
    name = 'crystalformer',
    version = '0.3',
    keywords='Crystal Generation',
    description = 'generative model for crystal structure',
    license = 'Apache License',
    #url = '',
    author = 'iopcompphys',
    author_email = 'zdcao@iphy.ac.cn, wanglei@iphy.ac.cn',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = [],
)