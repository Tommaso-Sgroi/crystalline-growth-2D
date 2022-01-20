#include <stdlib.h>
#include "../datastructures/dynamiclist.h"

#define RAND_SIZE_T() ({size_t retval;\
                          retval = 0;\
                        for (int i=0; i<64; i++) {\
                            retval = retval*2 + rand()%2;\
                        }\
                        retval;\
                    })

#define CHECK(call){                    \
    const cudaError_t error = call;     \
    if(error != cudaSuccess){           \
        printf("Error: %s:%d", __FILE__, __LINE__); \
        printf("code:%d, reason%s\n", error, cudaGetErrorString(error)); \
        exit(1);                        \
    }                                   \
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
	*seed = 6364136223846793005ULL * *seed + 1;
	return temper(*seed >> 16);
}

__device__ int position;
__global__ void sort_particles(particle* particles, int size){
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if(gloID >= size || particles[gloID].x < 0) return; //esce se il thread non ha particelle o non ha una particella valida
    
    position = 0;
    int pos = atomicAdd(&position, 1);
    particles[pos] = particles[gloID];

}
