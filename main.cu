#include "crystalline_growth/crystallinie_growth.cu"



int string_to_int(const char*);
void printDevProp(cudaDeviceProp devProp);

int main(int argc, const char* argv[]) {

    int x = string_to_int(argv[1]);
    int y = string_to_int(argv[2]);
    int iterazioni = string_to_int(argv[3]); 
    int numero_particelle = string_to_int(argv[4]);
    int posizione_seed_x = string_to_int(argv[5]);
    int posizione_seed_y = string_to_int(argv[6]);
    int write_out = string_to_int(argv[7]);


    return start_crystalline_growth(x, y, iterazioni, numero_particelle, posizione_seed_x, posizione_seed_y, write_out);

}



void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n",  devProp.major);
	printf("Minor revision number:         %d\n",  devProp.minor);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
	printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
	printf("Warp size:                     %d\n",  devProp.warpSize);
	printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
	printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
	printf("Maximum threads per SM:        %d\n",  devProp.maxThreadsPerMultiProcessor);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n",  devProp.clockRate);
	printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
	printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

int string_to_int(const char* c){
    int len = 0;
    if (1 == sscanf(c, "%i", &len)){
        return len;
    }
    return -1;
}
