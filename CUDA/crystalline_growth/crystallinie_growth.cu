#include "utility.h"

__global__ void move_and_precrystalize(particle* g_particles, particle* g_vect_precrystal, int * g_matrix, int len_x, int len_y, 
                                       int numero_particelle, int* g_numero_particelle_output){
    int locId = threadIdx.x;
    int gloID = get_globalId();

    if(gloID >= numero_particelle) return;  //i thread in più vengono scartati

    __shared__ int s_crystallized;
    if(locId == 0)
        s_crystallized = 0;     //inizializzo variabile nella shared per contare il numero di precristalli in un blocco
    __syncthreads();

    particle p = g_particles[gloID];
    if(check_crystal_neighbor(g_matrix, &p, len_x, len_y) == false){ // if non è stato precristallizato 

        int x = (lcg64_temper_p(&p) % 3) - 1;
        int y = (lcg64_temper_p(&p) % 3) - 1;

        int x_movement =  p.x + x; // pick random x direction
        int y_movement =  p.y + y; // pick random y direction
        if(!is_in_bounds(x_movement, y_movement, len_x, len_y)){
            x_movement = p.x;
            y_movement = p.y;
        }
        p.x = x_movement;
        p.y = y_movement;
        g_particles[gloID] = p;     //nuova posizione della particella 

    }
    else{
        g_vect_precrystal[gloID] = p;         //salvo particella da precristallizzare 
        atomicAdd(&s_crystallized, 1);        //aumento il contatore dei precristalli per blocco
        g_particles[gloID].x = -1;            // invalida particella
    }
    __syncthreads();

    if(locId == 0){ 
        atomicAdd(g_numero_particelle_output, s_crystallized);//salva numero di precristalli nella globale
    }
}


__global__ void crystallize(particle* g_vect_precrystal, int* g_matrix, int len_y, int size){
    int gloID = get_globalId();

    if(gloID >= size || g_vect_precrystal[gloID].x < 0) return; //i thread in più o quelli che non hanno un precristallo da cristallizzare vengono scartati
    particle p = g_vect_precrystal[gloID];
    g_matrix[p.x * len_y + p.y] = 1;        //aggiungo il precristallo alla matrice cioè creo un nuovo cristallo

    p.x = -1;                      
    g_vect_precrystal[gloID] = p; //invalido la particella sul vettore
}



__global__ void build_vector_particle(particle* g_particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = get_globalId();

    if(gloID >= numero_particelle) return;
    int rng_seed = (7 + gloID) * (7 * gloID + 1);
    particle p;
    do
    {
        //genera casualmente particelle nei limiti della matrice 
        p.x = lcg64_temper_i(rng_seed++) % len_x;
        p.y = lcg64_temper_i(rng_seed++) % len_y;
    }while(p.x == posizione_seed_x && p.y == posizione_seed_y);
    p.rng = lcg64_temper_i(rng_seed);
    g_particles[gloID] = p;             //aggiungo la particella al vettore in base al global id thread
}


__global__ void resize_vector_particles(particle* particles, particle* buffer, int numero_particelle, int* index){
    int gloID = get_globalId();

    if(gloID >= numero_particelle || particles[gloID].x < 0) return;

    int i = atomicAdd(index, 1);
    buffer[i] = particles[gloID];   //copio le particelle in un nuovo vettore in modo da eliminare i buchi formati dai precristalli

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
    
    transform_2D_space_in_1D_array(&h_space, h_buffer_field);       //serve per trasformare la matrice in vettore in modo da allocarla nella global memory

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

    for(int h_i = 0; h_i < h_iterazioni && h_numero_particelle > 0; h_i++){
        move_and_precrystalize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(
                d_vect_particle, d_vect_precrystal, d_matrix, h_x, h_y, h_numero_particelle, d_h_crystallized_particles_n);
        CHECK(cudaDeviceSynchronize());
        
        crystallize<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>>(d_vect_precrystal, d_matrix, h_y, h_numero_particelle);
        CHECK(cudaDeviceSynchronize());
        if(*d_h_crystallized_particles_n == 0) continue;    //se non si sono creati precristalli 
        CHECK(cudaMemset(d_index, 0, sizeof(int)));
        
        resize_vector_particles<<<get_grid_size(h_numero_particelle, H_NUM_THREAD), H_NUM_THREAD>>> (
            d_vect_particle, d_buffer, h_numero_particelle, d_index);
        CHECK(cudaDeviceSynchronize());

        h_numero_particelle -= *d_h_crystallized_particles_n;       //particelle rimanenti
        *d_h_crystallized_particles_n = 0;

        //scambio i puntatori ai vettori delle particelle in modo da evitare copie del vettore
        particle* tmp = d_vect_particle;
        d_vect_particle = d_buffer;
        d_buffer = tmp;
    }

    if(h_write_out == 1){
        CHECK(cudaMemcpy(h_buffer_field, d_matrix, h_x * h_y * sizeof(int), cudaMemcpyDeviceToHost));
        transfer_output(&h_space, h_buffer_field);      //trasforma la matrice caricata sulla global come vettore in matrice
        write_output(&h_space);         //stampa
    }

    CHECK(cudaFree(d_h_crystallized_particles_n));
    CHECK(cudaFree(d_matrix));
    CHECK(cudaFree(d_vect_particle));
    return 0;
}