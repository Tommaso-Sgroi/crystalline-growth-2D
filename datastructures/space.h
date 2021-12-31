#include "../crystalline_growth/utility.h"

#include <stdio.h>
#include <stdbool.h>

//_______________________________PARTICLE_______________________________________-
typedef struct {
    size_t x, y;
}particle;

/*Alloca nell'heap una nuova particella*/
particle* new_particle(){
        return malloc(sizeof(particle));
}

void print_particle(particle* p){
        printf("Paticle at: (%zu, %zu)\n", p->x, p->y);
}

//__________________________SPACE__________________________________
struct space{
        short** field;
        size_t len_x, len_y;
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


 inzializzo il campo (la matrice che contiene le shorte per le partishorte)
*/
/*short** build_field(const size_t x, const size_t y){

        short** field = (short**) calloc(x, sizeof(short*));
        for(size_t i = 0; i < x; i++){
                field[i] = (short*) calloc(y, sizeof(short)); // TODO PARALLELIZZARE
        }
        return field;
}*/

void build_field(struct space* space)
{
        space->field = (short**) calloc(space->len_x, sizeof(short*));
        for(size_t i = 0; i < space->len_x; i++){
                space->field[i] = (short*) calloc(space->len_y, sizeof(short)); // TODO PARALLELIZZARE
        }

};


void init_field(struct space* space, const size_t posizione_seed_x, const size_t posizione_seed_y/*, const size_t numero_partishorte*/){ 
       for(size_t x = 0; x < space->len_x; x++){
                for(size_t y = 0; y < space->len_y; y++){
                        space->field[x][y] = -1; // inizializzo la shorta
                }
        space->field[posizione_seed_x][posizione_seed_y] = 1;                       
       }
}




#define IS_IN_BOUNDS(x, y, len_x, len_y) (x >= 0 && x < len_x && y >= 0 && y < len_y) // poi voglio vedere se e cosa cambia utilizzando questa macro o la funzione

bool is_in_bounds(const size_t x, const size_t y, const size_t len_x, const size_t len_y){
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

bool check_crystal_neighbor(struct space* space, particle* p){
        static short points []= {-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1};

        for (int i = 0; i < 8; i++){
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
    for(size_t i = 0; i < s.len_x; i++){
        for(size_t j = 0; j < s.len_y; j++){
        char* c;
        if(s.field[i][j] == -1){
                c = " P";
        }
        if(s.field[i][j] == 0){
                c = "PC";
        }
        if(s.field[i][j] == 1){
                c = " C";
        }
        printf("%s%i ", c, s.field[i][j]);
        }
        printf("\n");
    }
        
}
