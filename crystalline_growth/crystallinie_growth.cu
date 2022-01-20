#include "utility.h"




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


__global__ void move_and_precrystalize(particle* particles, int * dev_matrix, int len_x, int len_y, 
                                       int iterazioni, int numero_particelle, int* numero_particelle_output){

    bool precrystal = false;
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int locID = threadIdx.x;
    // int GridSize = gridDim.x * blockDim.x;
    
    //if(gloID >= numero_particelle) return;

    __shared__ int crystallized[1];
    crystallized[0] = 0;

    int rng_seed = gloID;
    //for(int i = gloID; i < numero_particelle; i += GridSize){
    int i = gloID;
    //printf("GLOBAL_ID: %i\n", gloID);

    particle p = particles[i];                                          //particella
    if(gloID < numero_particelle){
        
        precrystal = check_crystal_neighbor(&dev_matrix, &p, len_x, len_y);
        if(!precrystal){                                                    // if non è stato precristallizato nulla

            int x_movement;
            int y_movement;

            printf("Muovo particella %i, %i\n", p.x, p.y);
            int x = (rand_lfsr113_Bits(rng_seed)%2 * (rand_lfsr113_Bits(rng_seed)%2? 1: -1));
            int y = (rand_lfsr113_Bits(rng_seed)%2 * (rand_lfsr113_Bits(rng_seed)%2? 1: -1));
            printf("Choosed: %i, %i\n", x, y);

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
            printf("Cristallizzo particella %i, %i\n", p.x, p.y);

            dev_matrix[p.x * len_y + p.y] = 1;
            atomicAdd(&crystallized[0], 1U);
            particles[i].x = -1;
        }
        __syncthreads();
        printf("Crisallizzate %i\n", crystallized[0]);
    }
    if(locID == 0){    
        atomicAdd(numero_particelle_output, crystallized[0]);
        printf("Crisallizzate GLOBALI %i\n", *numero_particelle_output);
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


int start_crystalline_growth(const int x, const int y, const int iterazioni, int numero_particelle,
           const int posizione_seed_x, const int posizione_seed_y, int write_out){
//----------------------------------------------------------------------------------------------
    struct space space;
    space.len_x=x;
    space.len_y=y;

    //costruisco il campo
    build_field(&space);
    init_field(&space, posizione_seed_x, posizione_seed_y);

    static const int NUM_THREAD = 256;
    int* dev_matrix;
    int* crystallized_particles_n;
    particle* dev_vect_particle;

    int buffer_field [x * y];
    for(int x1 = 0; x1 < x; x1++){
        for(int y1 = 0; y1 < y; y1++){
                buffer_field[x1 * y + y1] = space.field[x1][y1]; // inizializzo la inta
        }
    }


    int grid = (numero_particelle + NUM_THREAD - 1)/NUM_THREAD;
    CHECK(cudaMalloc((void**) &dev_matrix, x * y * sizeof(int)));
    CHECK(cudaMalloc((void**) &dev_vect_particle, numero_particelle * sizeof(particle)));
    CHECK(cudaMallocManaged(&crystallized_particles_n, sizeof(int)));


    CHECK(cudaMemcpy(dev_matrix, buffer_field, x * y * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(crystallized_particles_n, 0, 1));
 


    build_vector_particle<<< grid, NUM_THREAD  >>>(dev_vect_particle, numero_particelle, x, y, posizione_seed_x, posizione_seed_y);
    print_field_device<<<1,1>>>(dev_matrix, x, y);
    
    print_vet_particle<<<(numero_particelle+NUM_THREAD-1)/NUM_THREAD, NUM_THREAD>>>(dev_vect_particle, numero_particelle);
    CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < iterazioni && numero_particelle > 0; i++){
        move_and_precrystalize<<<(numero_particelle+NUM_THREAD-1)/NUM_THREAD, NUM_THREAD>>>(dev_vect_particle, dev_matrix, x, y, iterazioni, numero_particelle, crystallized_particles_n);
        CHECK(cudaDeviceSynchronize());
        
       
        printf("Numero particelle %i -> ", numero_particelle);
               
        numero_particelle -= *crystallized_particles_n; //aggiornamento delle particelle
        crystallized_particles_n[0] = 0;
        sort_particles<<<(numero_particelle + NUM_THREAD - 1)/ NUM_THREAD, NUM_THREAD>>>(dev_vect_particle, numero_particelle);
        CHECK(cudaDeviceSynchronize());

        printf("numero particelle dopo aggiornamento %i\n", numero_particelle);
       
    }
    print_field_device<<<1,1>>>(dev_matrix, x, y);


    CHECK(cudaDeviceSynchronize());
    if(write_out == 1){
        int space_monodimention[x * y];
        CHECK(cudaMemcpy(space_monodimention, dev_matrix, x * y * sizeof(int), cudaMemcpyDeviceToHost));
        transfer_output(&space, space_monodimention);
    }

    CHECK(cudaFree(crystallized_particles_n));
    CHECK(cudaFree(dev_matrix));
    CHECK(cudaFree(dev_vect_particle));
    return write_out == 1? write_output(&space): 0;
}