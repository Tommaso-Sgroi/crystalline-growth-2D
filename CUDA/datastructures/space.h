#include <stdio.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

//_______________________________PARTICLE_______________________________________-
typedef struct __align__(8) {
    int x, y, rng;
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

//costruisce matrice
void build_field(struct space* space){
        space->field = (int**) calloc(space->len_x, sizeof(int*));
        for(int i = 0; i < space->len_x; i++){
                space->field[i] = (int*) calloc(space->len_y, sizeof(int));
        }
}


void init_field(struct space* space, const int posizione_seed_x, const int posizione_seed_y){ 
       for(int x = 0; x < space->len_x; x++){
                for(int y = 0; y < space->len_y; y++){
                        space->field[x][y] = 0;
                }
        space->field[posizione_seed_x][posizione_seed_y] = 1;                       
       }
}

//controlla se la particella si muove nei limiti della matrice
__device__ bool is_in_bounds(const int x, const int y, const int len_x, const int len_y){       
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

//controlla se la particella ha un cristallo vicino
__device__ bool check_crystal_neighbor(int* space, particle* p, int len_x, int len_y){
        int points []= {-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1};
        int flag = false;
        for (int i = 0; i < 16; i++){
                int dx = points[i];
                int dy = points[++i];

                int new_x = p->x + dx;
                int new_y = p->y + dy;

                if(is_in_bounds(new_x, new_y, len_x, len_y) && 
                        space[new_x * len_y + new_y] == 1){
                        flag = true;
                }
        }
        return flag;
}

