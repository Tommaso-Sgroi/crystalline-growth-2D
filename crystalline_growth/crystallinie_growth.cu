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



void move_and_precrystalize(arraylist* particles, arraylist* precrystalize, struct space* space, int iterazioni){

    for(int k = 0; k < iterazioni; k++){
        if(particles->used > 0){
            for(int i = particles->used - 1 ;; i--){
                particle p = particles->array[i];
                // print_particle(p);
                
                if(check_crystal_neighbor(space,  &p)){
                    insertArray(precrystalize,  &p);
                    removeAt(particles, i);
                    //printf("Trovato cristallo vicino!\n");
                }

                else{
                    int x_movement;
                    int y_movement;

                    // printf("Muovo particella n\n");
                    do{
                        int x = 0;
                        int y = 0;
                        while (((x == 0) && (y == 0))){
                            x = (rand()%2 * (rand()%2? 1: -1));
                            y = (rand()%2 * (rand()%2? 1: -1));
                        }
                        
                        //printf("Choosed: %i, %i\n", x, y);

                        x_movement =  p.x + x; // pick random x direction
                        y_movement =  p.y + y; // pick random y direction
                    }while(!is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)); // finché non sceglie una direzione corretta continua a scegliere randomicamente
                                                                                                // sostituibile con (!_movement | y_movement)
                     p.x = x_movement;
                     p.y = y_movement;

                     //printf("Si muove in (%i, %i)\n", p->x, p->y);

                }
                //printf("\n");
                // lo scopo di questo if è puramente per uno scopo di limitazione dei numeri unsigned
                // quando viene decrementato e si trova a 0 va al numero massimo che può rappresentare
                // quindi questo controllo serve a evitare cose brutte
                if(i == 0) {
                    break; 
                }
            }
            // cristallizza
            for(int i = 0; i < precrystalize->used; i++){
                // printf("%i, %i\n", i, precrystalize->used);
                space->field[precrystalize->array[i].x][precrystalize->array[i].y] = 1;
            }
            
            if(precrystalize->used > 0){
                precrystalize->used = 0;
            }

            // print_grid(space, particles, precrystalize);
            // print_grid(space, particles);

            // printf("\n");
        }
        else{
            break;
        }
        // printf("\n");
    }
}


__global__ void build_vector_particle(particle* particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int GridSize = gridDim.x * blockDim.x;
    for(int i = gloID; i < numero_particelle; i += GridSize){
        particle info;
        do
        {
            info.x = rand_lfsr113_Bits(gloID + GridSize) % len_x;
            info.y = rand_lfsr113_Bits(gloID + GridSize) % len_y;
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
            printf("%i ", device_matrix[_x * len_y + _y]);
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
    particle* dev_vet_particle;

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


    cudaMalloc((void**) &dev_matrix, x * y * sizeof(int));
    cudaMemcpy(dev_matrix, buffer_field, x * y * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_vet_particle, numero_particelle * sizeof(particle));
    build_vector_particle<<<  (numero_particelle+numThread+1)/numThread, numThread  >>>(dev_vet_particle, numero_particelle, x, y, posizione_seed_x, posizione_seed_y);
    //print_field_device<<<1,1>>>(dev_matrix, x, y);
    //print_vet_particle<<<(numero_particelle+numThread+1)/numThread, numThread>>>(dev_vet_particle, numero_particelle);
    //numero_particelle+numThread+1/numThread, numThread
    //cudaMalloc((void**) &dev_matrix, x * y * sizeof(int));

    //inizializzo campo
    
    //costruisco vettore delle particelle in movimento (random)
    //build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, posizione_seed_x, posizione_seed_y);
    
    ///fino qui CPU

    /// da qui GPU

    //muovo e precristallizzo
    //move_and_precrystalize(&particles, &precrystallized_particles, &space, iterazioni);



    cudaDeviceSynchronize();
    return write_out == 1? write_output(&space): 0;
}