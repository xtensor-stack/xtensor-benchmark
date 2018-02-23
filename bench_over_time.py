import subprocess as sp
# import tinydb
import os
import shutil

absdir = os.path.dirname(os.path.realpath(__file__))

CONDA_PREFIX = None
XTENSOR_VERSION = [0, 0, 0]

def call(arguments):
    print(arguments)
    os.system(" ".join(arguments))

def parse_version():
    global XTENSOR_VERSION
    os.chdir(absdir + '/checkouts/xtensor')
    with open(absdir + '/checkouts/xtensor/include/xtensor/xtensor_config.hpp') as f:
        for line in f.readlines():
            if line.startswith("#define XTENSOR_VERSION_MAJOR"):
                XTENSOR_VERSION[0] = int(line.split()[-1])
            if line.startswith("#define XTENSOR_VERSION_MINOR"):
                XTENSOR_VERSION[1] = int(line.split()[-1])
            if line.startswith("#define XTENSOR_VERSION_PATCH"):
                XTENSOR_VERSION[2] = int(line.split()[-1])
    print(f"Testing against: {XTENSOR_VERSION}")

def init():
    global CONDA_PREFIX
    try:
        call(['conda', 'env', 'create'])
    except:
        pass
    try:
        shutil.rmtree(absdir + '/stats')
    except:
        pass
    call(['source', 'activate', 'xtensor-benchmark'])
    CONDA_PREFIX = os.environ['CONDA_PREFIX']
    print(f"CONDA PREFIX SET TO: {CONDA_PREFIX}")

def install_xtensor_version(version):
    try:
        os.mkdir(absdir + '/checkouts')
        call(['git', 'clone', 'https://github.com/QuantStack/xtensor'])
    except:
        os.chdir(absdir + '/checkouts/xtensor')
        call(['git', 'checkout', 'master'])
        call(['git', 'pull'])
    call(['git', 'checkout', version])

    parse_version()
    version_string = ".".join([str(el) for el in XTENSOR_VERSION])
    call(['conda', 'install', f'xtensor=={version_string}', '-c conda-forge', '-y'])

    try:
        os.mkdir(absdir + '/checkouts/xtensor/build')
    except:
        pass

    os.chdir(absdir + '/checkouts/xtensor/build')
    call(['cmake', '..', '-DCMAKE_INSTALL_LIBDIR=lib', f'-DCMAKE_INSTALL_PREFIX={CONDA_PREFIX}'])
    call(['make', 'install'])

def build_and_bench(version):
    try:
        os.mkdir(absdir + '/build')
    except:
        pass
    os.chdir(absdir + '/build')
    try:
        os.mkdir(absdir + '/stats')
    except:
        pass
    call(['cmake', '..', '-DBENCHMARK_EIGEN=ON'])
    call(['make', 'xpowerbench'])
    os.rename(absdir + '/build/bench.csv', absdir + f'/stats/{version}.csv')

def run():
    init()
    versions = ['master', '0.15.4', '0.14.0']
    for v in versions:
        install_xtensor_version(v)
        build_and_bench(v)

if __name__ == '__main__':
    run()