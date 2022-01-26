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
	*seed = 6364136223846793005ULL * (*seed) + 1;
	return temper(*seed >> 16);
}


__host__ int get_max_thread_x_block(){
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}



__global__ void print_particle_vector(particle* g_pv, int pn){
    for(int i = 0; i < pn; i++){
        print_particle(&g_pv[i]);
    }
}


__global__ void print_field(int* field, int len_x, int len_y){
    for(int i=0; i<len_x; i++){
        for(int j=0; j<len_y; j++){
            printf("%i ", field[i * len_y + j]);
        }
        printf("\n");
    }
    
}



__global__ void odd_even_sort(particle* g_p, int pn, int phase, int* has_swapped){
    int gloid = get_globalId();
    if(gloid >= pn) return;
    
    int start = gloid * 2 + phase;
    int end = start + 1;

    if(g_p[start].x < g_p[end].x){
        particle tmp = g_p[start];
        g_p[start] = g_p[end];
        g_p[end] = tmp;
        *has_swapped = 1;
    }
}


__host__ void sort_particles(particle* d_p, int h_pn, int h_tn){
    int h_phase = 0;
    int exit_fun = 0;
    int* d_h_has_swapped;
    CHECK(cudaMallocManaged(&d_h_has_swapped, sizeof(int)));
    *d_h_has_swapped = 0;
    
    for(int i = 0; i < h_pn; i++){

        odd_even_sort <<<get_grid_size(h_pn, h_tn), h_tn>>>(d_p, h_pn / 2, h_phase, d_h_has_swapped);
        CHECK(cudaDeviceSynchronize());

        if(*d_h_has_swapped > 0){
            exit_fun++;
        }
        else{
            exit_fun = 0;
            *d_h_has_swapped = 0;
        }

        if(*d_h_has_swapped >= 2) return;
        
        h_phase = (h_phase + 1) % 2;
    }
}