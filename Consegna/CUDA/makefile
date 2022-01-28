CC=nvcc
COUT=-o
CDEBUGFLAG=-g

main: main.cu
	$(CC) $(COUT) out/main.o main.cu

main_singlecore: main_singlecore.cu
	$(CC) $(COUT) out/main_singlecore.o main_singlecore.cu

main_debug: main.cu
	$(CC) $(CDEBUGFLAG) $(COUT) out/debug/main.db main.cu

.PHONY: clean
clean:
	find out -maxdepth 5 -type f -delete
