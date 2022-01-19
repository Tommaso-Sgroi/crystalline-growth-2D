#include "utility.h"



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
    }
}

// arraylist particles_movement, precrystalized_particles;

// static int len_x, len_y;


//struct space space;

//arraylist particles, precrystallized_particles;
void print_grid(struct space* s, arraylist* p){
        int grid [s->len_x][s->len_y];

        for(int x = 0; x < s->len_x; x++){
            for (int y = 0; y < s->len_y; y++){
                grid[x][y] = s->field[x][y];
            }
        }

        for(int i = 0; i < p->used; i++){
            grid[p->array[i].x][p->array[i].y] = -2;
        }
        

    for(int i = 0; i < s->len_x; i++){
        for(int j = 0; j < s->len_y; j++){

            if(grid[i][j] == -2){
                printf("P ");
            }
            if(grid[i][j] == -1){
                printf("0 ");
            }
            if(grid[i][j] == 1){
                printf("C ");
            }
            }
        printf("\n");
    }
}


__global__ void move_and_precrystalize(particle* particles, int * dev_matrix, int len_x, int len_y, int iterazioni, int numero_particelle){

    bool precrystal = false;
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int GridSize = gridDim.x * blockDim.x;

    int rng_seed = gloID;
    for(int i = gloID; i < numero_particelle; i += GridSize){
        
        particle p = particles[i];                                          //particella
        
        precrystal = check_crystal_neighbor(&dev_matrix, &p, len_x, len_y);
        if(!precrystal){                                                    // if non è stato precristallizato nulla

            int x_movement;
            int y_movement;

            // printf("Muovo particella n\n");
            int x = (rand_lfsr113_Bits(rng_seed)%2 * (rand_lfsr113_Bits(rng_seed)%2? 1: -1));
            int y = (rand_lfsr113_Bits(rng_seed)%2 * (rand_lfsr113_Bits(rng_seed)%2? 1: -1));
                //printf("Choosed: %i, %i\n", x, y);

            x_movement =  p.x + x; // pick random x direction
            y_movement =  p.y + y; // pick random y direction
            if(!is_in_bounds(x_movement, y_movement, len_x, len_y)){
                x_movement = p.x;
                y_movement = p.y;
            }
            p.x = x_movement;
            p.y = y_movement;
        }
        __syncthreads();  // controllato se ha finito il blocco allora passa alla cristallizzazione
        if(precrystal){
            dev_matrix[p.x * len_y + p.y] = 1;
        }
    }
}


__global__ void build_vector_particle(particle* particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int GridSize = gridDim.x * blockDim.x;
    for(int i = gloID; i < numero_particelle; i += GridSize){
        particle info;
        do
        {
            info.x = rand_lfsr113_Bits(gloID) % len_x;
            info.y = rand_lfsr113_Bits(gloID) % len_y;
        }while(info.x == posizione_seed_x && info.y == posizione_seed_y);
        particles[i] = info;
    }

}

__global__ void print_vet_particle(particle* particles, int numero_particelle) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int GridSize = gridDim.x * blockDim.x;
    for(int i = gloID; i < numero_particelle; i += GridSize){
        print_particle(&particles[i]);
    }
}


__global__ void print_field_device(int* device_matrix, int len_x, int len_y){

    for(int _x = 0; _x < len_x; _x++){
        for(int _y = 0; _y < len_y; _y++){
            printf("%s ", device_matrix[_x * len_y + _y] == 1? "C": "0");
        }
        printf("\n");
    }

}


int start_crystalline_growth(const int x, const int y, const int iterazioni, const int numero_particelle,
           const int posizione_seed_x, const int posizione_seed_y, int write_out){
//----------------------------------------------------------------------------------------------
    struct space space;
    space.len_x=x;
    space.len_y=y;

    //arraylist particles, precrystallized_particles;

    // printf("x: %i\ny: %i\nIterazioni: %i\nNumero particelle: %i\nPosizione seed: (%i, %i)\n",
    //     x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);

    //inizializzo la lista delle particelle

    //costruisco il campo
    build_field(&space);
    init_field(&space, posizione_seed_x, posizione_seed_y);

    int numThread=256;
    int* dev_matrix;
    particle* dev_vect_particle;

    // print_field(&space);
    // printf("\n");
    int buffer_field [x * y];
    for(int _x = 0; _x < x; _x++){
        for(int _y = 0; _y < y; _y++){
                buffer_field[_x * y + _y] = space.field[_x][_y]; // inizializzo la inta
                // printf("%i ", buffer_field[_x * y + _y]);
        }
        //printf("\n");
    }
    int block = numThread;
    int grid = (numero_particelle+block-1)/block;
    cudaMalloc((void**) &dev_matrix, x * y * sizeof(int));
    cudaMemcpy(dev_matrix, buffer_field, x * y * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_vect_particle, numero_particelle * sizeof(particle));
    build_vector_particle<<< grid, block  >>>(dev_vect_particle, numero_particelle, x, y, posizione_seed_x, posizione_seed_y);
    print_field_device<<<1,1>>>(dev_matrix, x, y);
    print_vet_particle<<<(numero_particelle+numThread-1)/numThread, numThread>>>(dev_vect_particle, numero_particelle);
    for(int i = 0; i < iterazioni; i++){
        move_and_precrystalize<<<grid, block>>>(dev_vect_particle, dev_matrix, x, y, iterazioni, numero_particelle);
        cudaDeviceSynchronize();
    }
    print_field_device<<<1,1>>>(dev_matrix, x, y);
    //cudaMemcpy(dev_matrix, buffer_field, x * y * sizeof(int), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    return write_out == 1? write_output(&space): 0;
}