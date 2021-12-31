#include "../datastructures/dynamiclist.h"




// arraylist particles_movement, precrystalized_particles;

// static size_t len_x, len_y;


//struct space space;

//arraylist particles, precrystallized_particles;

void move_and_precrystalize(arraylist* particles, arraylist* precrystalize, struct space* space, int iterazioni){
       
        for(int k=0; k<iterazioni; k++){

                for(size_t i=0; i<particles->used; i++){
                        if(check_crystal_neighbor(&space, particles->array[i])){
                                insertArray(&precrystalize, particles->array[i]);
                                removeAt(&particles, i);
                                printf("Trovato cristallo vicino!\n");
                        }
                        else{
                                size_t x_movement;
                                size_t y_movement;

                                printf("Muovo particella n");
                                do{
                                        x_movement = particles->array[i]->x + (rand()%2 * (rand()%2? 1: -1)); // pick random x direction
                                        y_movement = particles->array[i]->y + (rand()%2 * (rand()%2? 1: -1)); // pick random y direction
                                }while(!is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)); // finché non sceglie una direzione corretta continua a scegliere randomicamente

                                particles->array[i]->x = x_movement;
                                particles->array[i]->y = y_movement;

                        }

                }

                for(size_t i=0; i<precrystalize->used; i++){
                        space->field[precrystalize->array[i]->x][precrystalize->array[i]->y]=1;
                }
                if(precrystalize->used>0)
                {
                        trim_list(&precrystalize);

                }
        }

}

int start_crystalline_growth(const size_t x, const size_t y, const size_t iterazioni, const size_t numero_particelle,
           const size_t posizione_seed_x, const size_t posizione_seed_y){

        struct space space;
        space.len_x=x;
        space.len_y=y;
        
        arraylist particles, precrystallized_particles;

        printf("x: %zu\ny: %zu\nIterazioni: %zu\nNumero particelle: %zu\nPosizione seed: (%zu, %zu)\n",
            x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);
        
        //inizializzo la lista delle particelle
        initArray(&particles, numero_particelle + 1);
        initArray(&precrystallized_particles, 10);
        
        
        //costruisco il campo 
        //space.field = build_field(x, y);
        build_field(&space);
        //inizializzo campo
        init_field(&space, posizione_seed_x, posizione_seed_y);
        //printo matrice iniziale (solo seed)
        print_field(&space);
        
        //costruisco vettore delle particelle in movimento (random)
        build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, posizione_seed_x, posizione_seed_y);

        //muovo e precristallizzo 
        move_and_precrystalize(&particles, &precrystallized_particles, &space, iterazioni);
        //printo matrice cristalli
        printf("\n\n");
        print_field(&space);

        printf("%zu\n", RAND_SIZE_T());

        // for(int x = 0; x < 50; x++){
        // short x_movement = rand()%2 * (rand()%2? 1: -1);
        // short y_movement = rand()%2 * (rand()%2? 1: -1);
        // printf("%i, %i\n", x_movement, y_movement);
        // }
       /*
       ALGORITMO 3
        Si divide l’algoritmo in 2 istanti di tempo:
        movimento e precristallizzazione
        cristallizzazione

        Controlla se una particella ha un cristallo vicino
        se sì precristallizza
        altrimenti muovi
        Cristallizza le particelle precristallizate

        */
        


        return 0;
}