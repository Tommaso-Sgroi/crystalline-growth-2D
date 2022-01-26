time ./build/main_singlecore.o 50 100 1000000 1000 25 50 0
time mpiexec -n 4 ./build/main_mpi.o 50 100 1000000 1000 25 50 0
