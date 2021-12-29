#include <stdio.h>
#include "../datastructures/cell.c"

cell** field;

void print_field(const size_t x, const size_t y){

    for(size_t i = 0; i < x; i++){
        for(size_t j = 0; j < y; j++){
                printf("%s%i ", field[i][j].status >= 0? " ": "", field[i][j].status);
        }
        printf("\n");
    }
        
}

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
void build_field(const size_t x, const size_t y){

        field = (cell**) calloc(x, sizeof(cell*));
        for(size_t i = 0; i < x; i++){
                field[i] = (cell*) calloc(y, sizeof(cell)); // TODO PARALLELIZZARE
        }
}


void init_field(const size_t len_x, const size_t len_y, const size_t posizione_seed_x, const size_t posizione_seed_y){
        for(size_t x = 0; x < len_x; x++){
                for(size_t y = 0; y < len_y; y++){
                        cell* c = &field[x][y]; // inizializzo la cella
                        c->particles = 0;
                        c->particles_moved_in = 0;
                        c->status = -1; 
                        c->x = x;
                        c->y = y;
                }
        }
        cell* seed = &field[posizione_seed_x][posizione_seed_y];
        seed->particles = 1;
        seed->status = 1;
        
}

int start_crystalline_growth(const size_t x, const size_t y, const size_t iterazioni, const size_t numero_particelle,
           const size_t posizione_seed_x, const size_t posizione_seed_y){

               
        printf("x: %zu\ny: %zu\nIterazioni: %zu\nNumero particelle: %zu\nPosizione seed: (%zu, %zu)\n",
            x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);
        
        build_field(x, y);
        init_field(x, y, posizione_seed_x, posizione_seed_y);

        // for iterazioni
        //      precristallizza
        //      muovi le particelle
        //      cristallizza le particelle
        
        print_field(x, y);
        


        return 0;
}