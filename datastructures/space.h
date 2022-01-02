#include "../crystalline_growth/utility.h"

#include <stdio.h>
#include <stdbool.h>

//_______________________________PARTICLE_______________________________________-
typedef struct {
    int x, y;
}particle;

/*Alloca nell'heap una nuova particella*/
particle* new_particle(){
        return malloc(sizeof(particle));
}

void print_particle(particle* p){
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

void build_field(struct space* space)
{
        space->field = (int**) calloc(space->len_x, sizeof(int*));
        for(int i = 0; i < space->len_x; i++){
                space->field[i] = (int*) calloc(space->len_y, sizeof(int)); // TODO PARALLELIZZARE
        }

};


void init_field(struct space* space, const int posizione_seed_x, const int posizione_seed_y/*, const int numero_partiinte*/){ 
       for(int x = 0; x < space->len_x; x++){
                for(int y = 0; y < space->len_y; y++){
                        space->field[x][y] = -1; // inizializzo la inta
                }
        space->field[posizione_seed_x][posizione_seed_y] = 1;                       
       }
}




#define IS_IN_BOUNDS(x, y, len_x, len_y) (x >= 0 && x < len_x && y >= 0 && y < len_y) // poi voglio vedere se e cosa cambia utilizzando questa macro o la funzione

bool is_in_bounds(const int x, const int y, const int len_x, const int len_y){
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

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
