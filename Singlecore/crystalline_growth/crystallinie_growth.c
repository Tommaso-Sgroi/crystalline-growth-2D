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



void move_and_precrystalize(arraylist* particles, arraylist* precrystalize, struct space* space, int iterazioni){

    for(int k = 0; k < iterazioni; k++){
        if(particles->used > 0){
            for (int i = particles->used-1 ; i >= 0; i--){
                particle p = particles->array[i];

                if(check_crystal_neighbor(space,  &p)){
                    insertArray(precrystalize,  &p);
                    removeAt(particles, i);
                }

                else{
                    int x_movement;
                    int y_movement;

                    do{
                        int x = 0;
                        int y = 0;
                        x = (lcg64_temper_p(&p) % 3) - 1;
                        y = (lcg64_temper_p(&p) % 3) - 1;
                        
                        x_movement =  p.x + x; // pick random x direction
                        y_movement =  p.y + y; // pick random y direction
                    }while(!is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)); // finché non sceglie una direzione corretta continua a scegliere randomicamente
                                                                                                // sostituibile con (!_movement | y_movement)
                    p.x = x_movement;
                    p.y = y_movement;

                    //particles->array[i] = p;
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
        }
        else{
            break;
        }
        // printf("\n");
    }
}


void build_vector_particle(arraylist* particles_final, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    particle seed = {posizione_seed_x, posizione_seed_y};
    for (int i = 0; i < numero_particelle; i++){
        int rng_seed = (7 + i) * (7 * i + 1);
        //printf("Seed: %i\n", rng_seed);
        particle p;
        do
        {
            p.x = lcg64_temper_i(rng_seed++) % len_x;
            p.y = lcg64_temper_i(rng_seed++) % len_y;
        }while(p.x == seed.x && p.y == seed.y);
        p.rng = lcg64_temper_i(rng_seed);
        insertArray(particles_final, &p);
    }
}


int start_crystalline_growth(const int x, const int y, const int iterazioni, const int numero_particelle,
           const int posizione_seed_x, const int posizione_seed_y, int write_out){

    struct space space;
    space.len_x=x;
    space.len_y=y;

    arraylist particles, precrystallized_particles;

    //inizializzo la lista delle particelle
    initArray(&particles, numero_particelle + 1);
    initArray(&precrystallized_particles, 10);


    //costruisco il campo
    //inizializzo campo
    //costruisco vettore delle particelle in movimento (random)
    build_field(&space);
    init_field(&space, posizione_seed_x, posizione_seed_y);
    build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, posizione_seed_x, posizione_seed_y);
    //print_array(&particles);
    //muovo e precristallizzo
    move_and_precrystalize(&particles, &precrystallized_particles, &space, iterazioni);

    return write_out == 1? write_output(&space): 0;
}