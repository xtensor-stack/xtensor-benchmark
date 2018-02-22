import subprocess as sp
# import tinydb
import os

absdir = os.path.dirname(os.path.realpath(__file__))

CONDA_PREFIX = None

def call(arguments):
    print(arguments)
    os.system(" ".join(arguments))

def init():
    global CONDA_PREFIX
    try:
        call(['conda', 'env', 'create'])
    except:
        pass
    call(['source', 'activate', 'xtensor-benchmark'])
    CONDA_PREFIX = os.environ['CONDA_PREFIX']
    print(f"CONDA PREFIX SET TO: {CONDA_PREFIX}")

def install_xtensor_version(version):
    try:
        os.mkdir(absdir + '/checkouts')
        os.chdir(absdir + '/checkouts')
        call(['git', 'clone', 'https://github.com/QuantStack/xtensor'])
    except:
        os.chdir(absdir + '/checkouts/xtensor')
    call(['git', 'checkout', version])

    try:
        os.mkdir(absdir + '/checkouts/xtensor/build')
    except:
        pass

    os.chdir(absdir + '/checkouts/xtensor/build')
    call(['cmake', '..', '-DCMAKE_INSTALL_LIBDIR=lib', f'-DCMAKE_INSTALL_PREFIX={CONDA_PREFIX}'])
    call(['make', 'install'])

def build_and_bench():
    os.chdir(absdir + '/build')
    call(['make', 'xbenchmark'])

def run():
    init()
    versions = ['master', '0.14.0']
    for v in versions:
        install_xtensor_version(v)
        build_and_bench()

if __name__ == '__main__':
    run()