#include "utility.h"


__device__  bool cannot_move(int gloid_greater_than_pn, int is_not_local_0_and_invalid){
    return gloid_greater_than_pn || is_not_local_0_and_invalid;
}   

__global__ void move_and_precrystalize(particle* g_particles, particle* g_vect_precrystal, int * g_matrix, int len_x, int len_y, 
                                       int numero_particelle, int* g_numero_particelle_output){
    __shared__ int s_crystallized;
    int locId = threadIdx.x;

    int gloID = get_globalId();
    if(cannot_move(gloID >= numero_particelle, g_particles[gloID].x < 0 && locId != 0)) return;
   // printf("GLO: %i, LOC: %i\n", gloID, locId);
    if(locId == 0) s_crystallized = 0;
    __syncthreads();

    particle p = g_particles[gloID];
    //if((threadIdx.x == 0 && p.x >= 0) || threadIdx.x > 0){
    if(p.x >= 0)
    if(check_crystal_neighbor(g_matrix, &p, len_x, len_y) == false){ // if non è stato precristallizato 

        int x = (lcg64_temper_p(&p) % 3) - 1;
        int y = (lcg64_temper_p(&p) % 3) - 1;

        int x_movement =  p.x + x; // pick random x direction
        int y_movement =  p.y + y; // pick random y direction
        if(!is_in_bounds(x_movement, y_movement, len_x, len_y)){
            x_movement = p.x;
            y_movement = p.y;
        }
        //particle pp = p;
        p.x = x_movement;
        p.y = y_movement;
        g_particles[gloID] = p;

        // if(*g_numero_particelle_output >= 19970){
        //     printf("%i, Paticle at: (%i, %i)--> (%i, %i)\n", *g_numero_particelle_output, p.x, p.y, pp.x, pp.y);
        // }

    }
    else{
        g_vect_precrystal[gloID] = p;         // salva partiella sulle precristallizzate
        atomicAdd(&s_crystallized, 1);
        //atomicAdd(g_numero_particelle_output, 1);        // incrementa contatore delle precristallizzate del blocco
                // incrementa contatore delle precristallizzate del blocco
        g_particles[gloID].x = -1;            // invalida particella
    }
    //}
    __syncthreads();

    if(locId == 0){ 
        atomicAdd(g_numero_particelle_output, s_crystallized);//salva numero di particelle cristallizate nella globale
    }
}


__global__ void crystallize(particle* g_vect_precrystal, int* g_matrix, int len_y, int size){
    int gloID = get_globalId();
    if(gloID >= size || g_vect_precrystal[gloID].x < 0) return; //esce se il thread non ha particelle o non ha una particella valida
    particle p = g_vect_precrystal[gloID];
    g_matrix[p.x * len_y + p.y] = 1;

    p.x = -1;                       //invalido x
    g_vect_precrystal[gloID] = p; //invalido la particella sul vettore
}



__global__ void build_vector_particle(particle* g_particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = get_globalId();
    if(gloID >= numero_particelle) return;
    int rng_seed = (7 + gloID) * (7 * gloID + 1);
    //printf("Seed: %i\n", rng_seed);
    particle p;
    do
    {
        //genera casualmente particelle nella matrice 
        p.x = lcg64_temper_i(rng_seed++) % len_x;
        p.y = lcg64_temper_i(rng_seed++) % len_y;
    }while(p.x == posizione_seed_x && p.y == posizione_seed_y);
    p.rng = lcg64_temper_i(rng_seed);
    g_particles[gloID] = p;
}


__global__ void resize_vector_particles(particle* particles, particle* buffer, int numero_particelle, int* index){
    int gloID = get_globalId();
    //printf("HERE\n");

    if(gloID >= numero_particelle || particles[gloID].x < 0) return;

    int i = atomicAdd(index, 1);
    //printf("Buffer %i: %i, %i\n", i, particles[gloID].x, particles[gloID].y);

    buffer[i] = particles[gloID];

}

__host__ int start_crystalline_growth(const int h_x, const int h_y, const int h_iterazioni, int h_numero_particelle,
           const int h_posizione_seed_x, const int h_posizione_seed_y, int h_write_out){

    struct space h_space;
    h_space.len_x = h_x;
    h_space.len_y = h_y;

    build_field(&h_space);              //costruisco il campo
    init_field(&h_space, h_posizione_seed_x, h_posizione_seed_y);

    static const int H_NUM_THREAD = 1024;
    int* d_matrix;
    int* d_h_crystallized_particles_n;
    int* d_index; 
    particle* d_vect_particle;
    particle* d_vect_precrystal;
    particle* d_buffer;


    int h_buffer_field[h_x * h_y];
    
    transform_2D_space_in_1D_array(&h_space, h_buffer_field);

    //malloc for global memory
    CHECK(cudaMalloc((void**) &d_matrix, h_x * h_y * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_index, sizeof(int)));
    CHECK(cudaMalloc((void**) &d_vect_particle, h_numero_particelle * sizeof(particle)));
    CHECK(cudaMalloc((void**) &d_vect_precrystal, h_numero_particelle * sizeof(particle)));
    CHECK(cudaMalloc((void**) &d_buffer, h_numero_particelle * sizeof(particle)));

    CHECK(cudaMallocManaged(&d_h_crystallized_particles_n, sizeof(int)));

    CHECK(cudaMemcpy(d_matrix, h_buffer_field, h_x * h_y * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_h_crystallized_particles_n, 0, sizeof(int)));
    CHECK(cudaMemset(d_vect_precrystal, -1, h_numero_particelle * sizeof(particle)));

    build_vector_particle<<< get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD  >>>(d_vect_particle, h_numero_particelle, h_space.len_x, h_space.len_y, h_posizione_seed_x, h_posizione_seed_y);
    CHECK(cudaDeviceSynchronize());

    //print_particle_vector<<<1,1>>>(d_vect_particle, h_numero_particelle);
    //CHECK(cudaDeviceSynchronize());

    for(int h_i = 0; h_i < h_iterazioni && h_numero_particelle > 0; h_i++){
        move_and_precrystalize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(
                d_vect_particle, d_vect_precrystal, d_matrix, h_x, h_y, h_numero_particelle, d_h_crystallized_particles_n);
        CHECK(cudaDeviceSynchronize());
        
        crystallize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(d_vect_precrystal, d_matrix, h_y, h_numero_particelle);
        CHECK(cudaDeviceSynchronize());
        // if(*d_h_crystallized_particles_n == h_numero_particelle)
        //     printf("%i\n", h_i);
        //printf("\n");
        CHECK(cudaMemset(d_index, 0, sizeof(int)));
        resize_vector_particles<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>> (
            d_vect_particle, d_buffer, h_numero_particelle, d_index
        );
        CHECK(cudaDeviceSynchronize());

        //printf("\n");
        h_numero_particelle -= *d_h_crystallized_particles_n;
        //printf("Cristallizzate: %i; Rimaste: %i\n", *d_h_crystallized_particles_n, h_numero_particelle);
        *d_h_crystallized_particles_n = 0;

        //CHECK(cudaMemcpy(d_vect_particle, d_buffer, h_numero_particelle * sizeof(particle), cudaMemcpyDeviceToDevice));
        //print_particle_vector<<<1,1>>>(d_vect_particle, h_numero_particelle);
        particle* tmp = d_vect_particle;
        d_vect_particle = d_buffer;
        d_buffer = tmp;
        //CHECK(cudaDeviceSynchronize());

        //printf("NUmero particelle %i\n", h_numero_particelle);
    }


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