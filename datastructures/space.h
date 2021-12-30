#include "../crystalline_growth/utility.h"
#include <stdio.h>
#include <stdbool.h>

//_______________________________PARTICELLE_______________________________________-
typedef struct {
    size_t x, y;
}particle;


//__________________________CELLE___________________________________
typedef struct {

    // size_t particles; // particelle che si tovano nella cella e si sono mosse O non possono muoversi // MUOVE DA QUA
    // size_t particles_moved_in; //particelle mosse in una cella // INSERISCE QUA
    
    short status; // -1 non è un cristallo // 0 precristallizzazione // 1 cristallo 
    
    size_t x; // probabilmente da rimuovere
    size_t y;// probabilmente da rimuovere

}cell;

void print_cell(cell* c){
        printf("Status: %i - Coordinate: (%zu,%zu)\n",
                c->status,             c->x, c->y);
}

//__________________________SPACE__________________________________
struct space{
        cell** field;
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


 inzializzo il campo (la matrice che contiene le celle per le particelle)
*/
cell** build_field(const size_t x, const size_t y){

        cell** field = (cell**) calloc(x, sizeof(cell*));
        for(size_t i = 0; i < x; i++){
                field[i] = (cell*) calloc(y, sizeof(cell)); // TODO PARALLELIZZARE
        }
        return field;
}

void init_field(struct space* space, const size_t len_x, const size_t len_y, const size_t posizione_seed_x, const size_t posizione_seed_y/*, const size_t numero_particelle*/){ 
        for(size_t x = 0; x < len_x; x++){
                for(size_t y = 0; y < len_y; y++){
                        space->field[x][y].status = -1; // inizializzo la cella
                }
        }
        space->field[posizione_seed_x][posizione_seed_y].status = 1;                       

}


#define IS_IN_BOUNDS(x, y, len_x, len_y) (x >= 0 && x < len_x && y >= 0 && y < len_y) // poi voglio vedere se e cosa cambia utilizzando questa macro o la funzione

bool is_in_bounds(size_t x, size_t y, size_t len_x, size_t len_y){
        return x >= 0 && x < len_x && y >= 0 && y < len_y;
}

bool check_crystal_neighbor(struct space* space, particle* p){
        static short points []= {-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1};

        for (int i = 0, x = 0; i < 8; i++, x++){
                int dx = points[i];
                int dy = points[++i];

                int new_x = p->x + dx;
                int new_y = p->y + dy;

                if(is_in_bounds(new_x, new_y, space->len_x, space->len_y) && 
                  space->field[new_x][new_y].status == 1){
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
                if(s.field[i][j].status == -1){
                        c = " P";
                }
                if(s.field[i][j].status == 0){
                        c = "PC";
                }
                if(s.field[i][j].status == 1){
                        c = " C";
                }
                printf("%s%i ", c, s.field[i][j].status);
        }
        printf("\n");
    }
        
}
