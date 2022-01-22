
#include <stdio.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

//_______________________________PARTICLE_______________________________________-
typedef struct __align__(8) {
    int x, y;
}particle;

/*Alloca nell'heap una nuova particella*/

__device__ void print_particle(particle* p){
        printf("Paticle at: (%i, %i)\n", p->x, p->y);
}

//__________________________SPACE__________________________________
struct space{
        int** field;
        int len_x, len_y;
};



/*
matrix:
  y y y y y y y y y 
 x
 x
 x
 x
 x
 x


 inzializzo il campo (la matrice che contiene le inte per le partiinte)
*/
/*int** build_field(const int x, const int y){

        int** field = (int**) calloc(x, sizeof(int*));
        for(int i = 0; i < x; i++){
                field[i] = (int*) calloc(y, sizeof(int)); // TODO PARALLELIZZARE
        }
        return field;
}*/

void build_field(struct space* space){
        space->field = (int**) calloc(space->len_x, sizeof(int*));
        for(int i = 0; i < space->len_x; i++){
                space->field[i] = (int*) calloc(space->len_y, sizeof(int)); // TODO PARALLELIZZARE
        }
}




void init_field(struct space* space, const int posizione_seed_x, const int posizione_seed_y/*, const int numero_partiinte*/){ 
       for(int x = 0; x < space->len_x; x++){
                for(int y = 0; y < space->len_y; y++){
                        space->field[x][y] = -1; // inizializzo la inta
                }
        space->field[posizione_seed_x][posizione_seed_y] = 1;                       
       }
}


// __global__ void print_vet_particle(particle* particles, int h_numero_particelle) {
//     int gloID = get_globalId();
//     int GridSize = gridDim.x * blockDim.x;
//     for(int i = gloID; i < h_numero_particelle; i += GridSize){
//         print_particle(&particles[i]);
//     }
// }


__global__ void print_field_device(int* device_matrix, int len_x, int len_y){

    for(int _x = 0; _x < len_x; _x++){
        for(int _y = 0; _y < len_y; _y++){
            printf("%s ", device_matrix[_x * len_y + _y] == 1? "C": "0");
        }
        printf("\n");
    }

}

__device__ bool is_in_bounds(const int x, const int y, const int len_x, const int len_y){
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

__device__ bool check_crystal_neighbor(int* space, particle* p, int len_x, int len_y){
        int points []= {-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1};
        
        for (int i = 0; i < 16; i++){
                int dx = points[i];
                int dy = points[++i];

                int new_x = p->x + dx;
                int new_y = p->y + dy;

                if(is_in_bounds(new_x, new_y, len_x, len_y) && 
                        space[new_x * len_y + new_y] == 1){
                        return true;
                }
        }
        return false;
}

