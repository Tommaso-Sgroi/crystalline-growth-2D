#include "../datastructures/dynamiclist.h"




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


void build_vector_particle(arraylist* particles, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    for (int i=0; i<numero_particelle; i++){
        particle info;
        do
        {
            info.x = rand() % len_x;
            info.y = rand() % len_y;
        }while(info.x == posizione_seed_x && info.y == posizione_seed_y);
        insertArray(particles, &info);
    }

}


int start_crystalline_growth(const int x, const int y, const int iterazioni, const int numero_particelle,
           const int posizione_seed_x, const int posizione_seed_y){

    struct space space;
    space.len_x=x;
    space.len_y=y;

    arraylist particles, precrystallized_particles;

    // printf("x: %i\ny: %i\nIterazioni: %i\nNumero particelle: %i\nPosizione seed: (%i, %i)\n",
    //     x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);

    //inizializzo la lista delle particelle
    initArray(&particles, numero_particelle + 1);
    initArray(&precrystallized_particles, 10);


    //costruisco il campo
    //space.field = build_field(x, y);
    build_field(&space);
    //inizializzo campo
    init_field(&space, posizione_seed_x, posizione_seed_y);
    //printo matrice iniziale (solo seed)
    // print_field(&space);
    // printf("\n\n");
    //costruisco vettore delle particelle in movimento (random)
    build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, posizione_seed_x, posizione_seed_y);
    
    // print_grid(&space, &particles);

    //muovo e precristallizzo
    move_and_precrystalize(&particles, &precrystallized_particles, &space, iterazioni);
    //printo matrice cristalli
    // printf("\n\n");
    print_field(&space);


    // for(int x = 0; x < 50; x++){
    // int x_movement = rand()%2 * (rand()%2? 1: -1);
    // int y_movement = rand()%2 * (rand()%2? 1: -1);
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