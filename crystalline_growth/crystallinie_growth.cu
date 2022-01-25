#include "utility.h"


__global__ void move_and_precrystalize(particle* g_particles, particle* g_vect_precrystal, int * g_matrix, int len_x, int len_y, 
                                       int iterazioni, int numero_particelle, int* g_numero_particelle_output, int seed){

    int gloID = get_globalId();
    if(gloID >= numero_particelle) return; // se il thread è fuori dal range delle particelle 

    int locID = threadIdx.x;
    int rng_seed = gloID + seed;

    __shared__ int s_crystallized;
    s_crystallized = 0;

    particle p = g_particles[gloID];                                          //particella
    if(check_crystal_neighbor(g_matrix, &p, len_x, len_y) == false){           // if non è stato precristallizato 

        int x_movement;
        int y_movement;

        int x = (lcg64_temper(&rng_seed) % 2 * (lcg64_temper(&rng_seed) % 2? 1: -1));
        int y = (lcg64_temper(&rng_seed) % 2 * (lcg64_temper(&rng_seed) % 2? 1: -1));

        x_movement =  p.x + x; // pick random x direction
        y_movement =  p.y + y; // pick random y direction
        if(!is_in_bounds(x_movement, y_movement, len_x, len_y)){
            x_movement = p.x;
            y_movement = p.y;
        }
        p.x = x_movement;
        p.y = y_movement;
        g_particles[gloID] = p;
    }
    else{
        g_vect_precrystal[gloID] = p;         // salva partiella sulle precristallizzate
        atomicAdd(&s_crystallized, 1);        // incrementa contatore delle precristallizzate del blocco
        g_particles[gloID].x = -1;            // invalida particella
    }
    __syncthreads();

    if(locID == 0){    
        atomicAdd(g_numero_particelle_output, s_crystallized); //salva numero di particelle cristallizate nella globale
    }



}


__global__ void crystallize(particle* g_vect_precrystal, int* g_matrix, int len_y, int size){
    int gloID = get_globalId();
    if(gloID >= size || g_vect_precrystal[gloID].x < 0) return; //esce se il thread non ha particelle o non ha una particella valida
    particle p = g_vect_precrystal[gloID];
    g_matrix[p.x * len_y + p.y] = 1;

    //printf("CRISTALLIZZO: %i, %i:", p.x, p.y);
    
    p.x = -1;                       //invalido x
    p.y = -1;                       //invalido y
    g_vect_precrystal[gloID] = p; //invalido la particella sul vettore
}



__global__ void build_vector_particle(particle* g_particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = get_globalId();
    if(gloID >= numero_particelle) return;
    int seed = gloID + 7;
    seed = ((gloID) * seed) + (seed * 13) >> 3;
    particle p;
    do
    {
        //genera casualmente particelle nella matrice 
        p.x = lcg64_temper(&seed) % len_x;
        p.y = lcg64_temper(&seed) % len_y;
    }while(p.x == posizione_seed_x && p.y == posizione_seed_y);
    g_particles[gloID] = p;
}



__host__ int start_crystalline_growth(const int h_x, const int h_y, const int h_iterazioni, int h_numero_particelle,
           const int h_posizione_seed_x, const int h_posizione_seed_y, int h_write_out){

    struct space h_space;
    h_space.len_x = h_x;
    h_space.len_y = h_y;

    //costruisco il campo
    build_field(&h_space);
    init_field(&h_space, h_posizione_seed_x, h_posizione_seed_y);


    static const int H_NUM_THREAD = get_max_thread_x_block();
    int* d_matrix;
    int* d_h_crystallized_particles_n;
    particle* d_vect_particle;
    particle* d_vect_precrystal;

    int h_buffer_field[h_x * h_y];
    transform_2D_space_in_1D_array(&h_space, h_buffer_field);

    //malloc for global memory
    CHECK(cudaMalloc((void**) &d_matrix, h_x * h_y * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_vect_particle, h_numero_particelle * sizeof(particle)));
    CHECK(cudaMalloc((void**) &d_vect_precrystal, h_numero_particelle * sizeof(particle)));

    CHECK(cudaMallocManaged(&d_h_crystallized_particles_n, sizeof(int)));

    CHECK(cudaMemcpy(d_matrix, h_buffer_field, h_x * h_y * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_h_crystallized_particles_n, 0, 1));
    CHECK(cudaMemset(d_vect_precrystal, -1, h_numero_particelle * sizeof(particle)));

    build_vector_particle<<< get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD  >>>(d_vect_particle, h_numero_particelle, h_space.len_x, h_space.len_y, h_posizione_seed_x, h_posizione_seed_y);
    CHECK(cudaDeviceSynchronize());

    
    for(int h_i = 0, h_seed = 0xEE234f12; h_i < h_iterazioni && h_numero_particelle > 0; h_i++, h_seed *= 7){
        move_and_precrystalize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(
                d_vect_particle, d_vect_precrystal, d_matrix, h_x, h_y, h_iterazioni, h_numero_particelle, d_h_crystallized_particles_n, h_seed
            );
        CHECK(cudaDeviceSynchronize());
        
        if(*d_h_crystallized_particles_n < 0) continue;

        crystallize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(d_vect_precrystal, d_matrix, h_y, h_numero_particelle);
        CHECK(cudaDeviceSynchronize());
        
        sort_particles(d_vect_particle, h_numero_particelle, H_NUM_THREAD);
        
        h_numero_particelle -= *d_h_crystallized_particles_n; //aggiornamento del numero di particelle restanti
        CHECK(cudaMemset(d_h_crystallized_particles_n, 0, 1));
    }
    //printf("NP: %i\n", h_numero_particelle);

    if(h_write_out == 1){
        CHECK(cudaMemcpy(h_buffer_field, d_matrix, h_x * h_y * sizeof(int), cudaMemcpyDeviceToHost));
        transfer_output(&h_space, h_buffer_field);
        write_output(&h_space);
    }

    CHECK(cudaFree(d_h_crystallized_particles_n));
    CHECK(cudaFree(d_matrix));
    CHECK(cudaFree(d_vect_particle));
    return 0;
}