
#include <stdio.h>
#include <stdbool.h>

//_______________________________PARTICLE_______________________________________-
typedef struct {
    int x, y, rng;
}particle;

/*Alloca nell'heap una nuova particella*/

void print_particle(particle* p){
        printf("Paticle at: (%i, %i)\n", p->x, p->y);
}

//__________________________SPACE__________________________________
struct space{
        int** field;
        int len_x, len_y;
};


//costruisce matrice
void build_field(struct space* space)
{
        space->field = (int**) calloc(space->len_x, sizeof(int*));
        for(int i = 0; i < space->len_x; i++){
                space->field[i] = (int*) calloc(space->len_y, sizeof(int)); 
        }

};


void init_field(struct space* space, particle seed){ 
       for(int x = 0; x < space->len_x; x++){
                for(int y = 0; y < space->len_y; y++){
                        space->field[x][y] = -1;
                }
        space->field[seed.x][seed.y] = 1;                       
       }
}

//controlla se la particella si muove nei limiti della matrice
bool is_in_bounds(const int x, const int y, const int len_x, const int len_y){
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

//controlla se la particella ha un cristallo vicino
bool check_crystal_neighbor(struct space* space, particle* p){
        static int points []= {-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1};
        for (int i = 0; i < 16; i++){
                int dx = points[i];
                int dy = points[++i];

                int new_x = p->x + dx;
                int new_y = p->y + dy;

                if(is_in_bounds(new_x, new_y, space->len_x, space->len_y) && 
                        space->field[new_x][new_y] == 1){
                        return true;
                }
        }
        return false;
}


void print_field(struct space* space){

    struct space s = *space;
    for(int i = 0; i < s.len_x; i++){
        for(int j = 0; j < s.len_y; j++){
                if(s.field[i][j] == -1){
                        printf("0 ");
                }
                if(s.field[i][j] == 1){
                        printf("C ");
                }
        }
        printf("\n");
    }
}
