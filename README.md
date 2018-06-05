# Benchmarks for linear algebra frameworks

This benchmarking suite allows a direct comparison of popular linear algebra frameworks in C++.
The libraries are easy-to-install using the conda package manager:

```
conda env create -f environment.yml
```

The environment.yml installs the following libraries:

- xtensor with xsimd
- Eigen3
- Armadillo
- Blitz++

After setting up the environment, it is advised to create a `build` directory, and execute `cmake`:

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBENCHMARK_ALL=ON
```

You should see a message for each found library, similar to the following:

```
          COMPILING WITH
======================================


Found eigen     : /home/myuser/miniconda3/envs/bench/include/eigen3
Found Blitz     : /home/myuser/miniconda3/envs/bench/include | /home/myuser/miniconda3/envs/bench/lib/libblitz.so
Found Armadillo : /home/myuser/miniconda3/envs/bench/include | /home/myuser/miniconda3/envs/bench/lib/libarmadillo.so
Found xtensor   : /home/myuser/miniconda3/envs/bench/include
Found xsimd     : /home/myuser/miniconda3/envs/bench/include
```

This allows you to make sure you're compiling with the correct, up-to-date versions of the libraries.

To build and run the benchmarks, just use the following command:

```
make xbenchmark
```

If you are only interested in specific benchmarks, build with `make xtensor_benchmark` and then run manually `./xtensor_benchmark --benchmark_filter=my_benchmark`. The backend to the benchmarks is the popular google-benchmark suite, so look there for more documentation.
