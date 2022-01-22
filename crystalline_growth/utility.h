#include <stdlib.h>
#include "../datastructures/space.h"


#define CHECK(call){                    \
    const cudaError_t error = call;     \
    if(error != cudaSuccess){           \
        printf("Error: %s:%d", __FILE__, __LINE__); \
        printf("code: %d, reason %s\n", error, cudaGetErrorString(error)); \
        exit(1);                        \
    }                                   \
}


__device__ int get_globalId(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__host__ int get_grid_size(int numero_particelle, const int NUM_THREAD){
    return (numero_particelle + NUM_THREAD - 1)/NUM_THREAD;
}

__host__ int write_output(struct space* s){
    FILE *f = fopen("output.space", "w");
    if (f == NULL){
        printf("Error opening file!\n");
        exit(1);
    }


    for(int x = 0; x < s->len_x; x++){
        for(int y = 0; y < s->len_y; y++){
            fprintf(f, "%s ", s->field[x][y] == 1? "C": "0");
        }
        fprintf(f, "%s", "\n");
    }
    fclose(f);
    return 0;
}


__host__ void transform_2D_space_in_1D_array(space* space, int* buffer_field){
    for(int x1 = 0; x1 < space->len_x; x1++){
        for(int y1 = 0; y1 < space->len_y; y1++){
            buffer_field[x1 * space->len_y + y1] = space->field[x1][y1];
        }
    }
}


__host__ void transfer_output(space* s, int* output_field){

    for(int x = 0; x < s->len_x; x++){
        for(int y = 0; y < s->len_y; y++){
            s->field[x][y] = output_field[x * s->len_y + y];
        }
    }
}


__device__ int temper(int x)
{
	x ^= x>>11;
	x ^= x<<7 & 0x9D2C5680;
	x ^= x<<15 & 0xEFC60000;
	x ^= x>>18;
	return x;
}


__device__ int lcg64_temper(int* seed){
    //printf("SEED_START: %i\n", *seed);

	*seed = 6364136223846793005ULL * (*seed) + 1;
    //printf("SEED_END: %i\n", *seed);

    int out = temper(*seed >> 16);
	return out;
}

__device__ int position;
__global__ void sort_particles(particle* particles, int size){
    int gloID = get_globalId();
    if(gloID >= size || particles[gloID].x < 0) return; //esce se il thread non ha particelle o non ha una particella valida
    
    position = 0;
    particle p = particles[gloID];
    int pos = atomicAdd(&position, 1);
    particles[pos] = p;

}



__host__ int get_max_thread_x_block(){
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}