#include "utility.h"


__global__ void move_and_precrystalize(particle* particles, particle* vect_precrystal, int * matrix, int len_x, int len_y, 
                                       int iterazioni, int numero_particelle, int* d_h_numero_particelle_output, int seed){

    int gloID = get_globalId();
    if(gloID >= numero_particelle) return; // scarta se fuori dal range

    bool precrystal = false;
    int locID = threadIdx.x;
    int rng_seed = gloID + seed;

    if(gloID == 0)
    {
        printf("GLOID: %i: RNG: %i; SEED: %i;\n", gloID, rng_seed, seed);
    }    
    __syncthreads();

    // int GridSize = gridDim.x * blockDim.x;
    
    //if(gloID >= h_numero_particelle) return;

    __shared__ int crystallized[1];
    crystallized[0] = 0;

    particle p = particles[gloID];                                          //particella
    //printf("RNG: %i\n", rng_seed);
    precrystal = check_crystal_neighbor(matrix, &p, len_x, len_y);
    if(!precrystal){           // if non è stato precristallizato nulla

        int x_movement;
        int y_movement;

        int x = (lcg64_temper(&rng_seed) % 2 * (lcg64_temper(&rng_seed) % 2? 1: -1));
        int y = (lcg64_temper(&rng_seed) % 2 * (lcg64_temper(&rng_seed) % 2? 1: -1));
        //printf("Muovo particella (%i, %i); SEED: %i; Choosed: %i, %i\n", p.x, p.y, rng_seed, x, y);

        x_movement =  p.x + x; // pick random x direction
        y_movement =  p.y + y; // pick random y direction
        if(!is_in_bounds(x_movement, y_movement, len_x, len_y)){
            //printf("IS_NOT_IN_BOUNDS\n");
            x_movement = p.x;
            y_movement = p.y;
        }
        p.x = x_movement;
        p.y = y_movement;
        particles[gloID] = p;
    }
    else{
        vect_precrystal[gloID] = p;         // salva partiella sulle precristallizzate
        atomicAdd(crystallized, 1);     // incrementa contatore delle precristallizzate del blocco
        particles[gloID].x = -1;            // invalida particella
    }
    if(gloID == 0)
        printf("RNG: %i; SEED: %i\n________\n", rng_seed, seed);
    __syncthreads();
    if(locID == 0){    
        atomicAdd(d_h_numero_particelle_output, crystallized[0]);
    }



}


__global__ void crystallize(particle* dev_vect_precrystal, int* dev_matrix, int len_y, int size){
    int gloID = get_globalId();
    if(gloID >= size || dev_vect_precrystal[gloID].x < 0) return; //esce se il thread non ha particelle o non ha una particella valida
    particle p = dev_vect_precrystal[gloID];
    dev_matrix[p.x * len_y + p.y] = 1;
    
    p.x = -1;                       //invalido x
    p.y = -1;                       //invalido y
    dev_vect_precrystal[gloID] = p; //invalido la particella sul vettore
}



__global__ void build_vector_particle(particle* particles, int h_numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = get_globalId();
    if(gloID >= h_numero_particelle) return;
    int seed = gloID + 7;
    seed = ((gloID) * seed) + (seed * 13) >> 3;
    particle p;
    do
    {
        p.x = lcg64_temper(&seed) % len_x;
        p.y = lcg64_temper(&seed) % len_y;
    }while(p.x == posizione_seed_x && p.y == posizione_seed_y);
    particles[gloID] = p;
}




__host__ int start_crystalline_growth(const int x, const int y, const int iterazioni, int h_numero_particelle,
           const int h_posizione_seed_x, const int h_posizione_seed_y, int h_write_out){
//----------------------------------------------------------------------------------------------
    struct space h_space;
    h_space.len_x=x;
    h_space.len_y=y;

    //costruisco il campo
    build_field(&h_space);
    init_field(&h_space, h_posizione_seed_x, h_posizione_seed_y);


    static const int H_NUM_THREAD = get_max_thread_x_block();
    int* d_matrix;
    int* d_h_crystallized_particles_n;
    int *d_position;
    particle* d_vect_particle;
    particle* d_vect_precrystal;

    int h_buffer_field[x * y];
    transform_2D_space_in_1D_array(&h_space, h_buffer_field);


    //int grid = (h_numero_particelle + H_NUM_THREAD - 1)/H_NUM_THREAD;
    CHECK(cudaMalloc((void**) &d_matrix, x * y * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_vect_particle, h_numero_particelle * sizeof(particle)));
    CHECK(cudaMalloc((void**) &d_vect_precrystal, h_numero_particelle * sizeof(particle)));
    CHECK(cudaMalloc((void**) &d_position, sizeof(int)));

    CHECK(cudaMallocManaged(&d_h_crystallized_particles_n, sizeof(int)));

    //printf("Allocata memoria\n");

    CHECK(cudaMemcpy(d_matrix, h_buffer_field, x * y * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_h_crystallized_particles_n, 0, 1));
    CHECK(cudaMemset(d_position, 0, 1));
    CHECK(cudaMemset(d_vect_precrystal, -1, h_numero_particelle * sizeof(particle)));

    //printf("Inizializzata memoria\n");
 

    build_vector_particle<<< get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD  >>>(d_vect_particle, h_numero_particelle, h_space.len_x, h_space.len_y, h_posizione_seed_x, h_posizione_seed_y);
    //printf("Coatruito vettore particelle\n");
    CHECK(cudaDeviceSynchronize());
    //printf("STart\n");

    for(int h_i = 0, h_seed = 1; h_i < iterazioni && h_numero_particelle > 0; h_i++, h_seed++){
        move_and_precrystalize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(
                d_vect_particle, d_vect_precrystal, d_matrix, x, y, iterazioni, h_numero_particelle, d_h_crystallized_particles_n, h_seed
            );
        CHECK(cudaDeviceSynchronize());
        
        crystallize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(d_vect_precrystal, d_matrix, y, h_numero_particelle);
        CHECK(cudaDeviceSynchronize());
       
        //printf("Numero particelle %i -> ", h_numero_particelle);
        sort_particles<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(d_vect_particle, d_position, h_numero_particelle);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_position, 0, 1));
        
        h_numero_particelle -= *d_h_crystallized_particles_n; //aggiornamento delle particelle
        CHECK(cudaMemset(d_h_crystallized_particles_n, 0, 1));
    }

    if(h_write_out == 1){
        CHECK(cudaMemcpy(h_buffer_field, d_matrix, x * y * sizeof(int), cudaMemcpyDeviceToHost));
        transfer_output(&h_space, h_buffer_field);
        write_output(&h_space);
    }

    CHECK(cudaFree(d_h_crystallized_particles_n));
    CHECK(cudaFree(d_matrix));
    CHECK(cudaFree(d_vect_particle));
    CHECK(cudaFree(d_position));
    return 0;
}