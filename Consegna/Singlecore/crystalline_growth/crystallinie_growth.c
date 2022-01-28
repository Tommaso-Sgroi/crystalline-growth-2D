#include "utility.h"

//muovo tutte le particelle e precristallizzo quelle vicini ad un cristallo
void move_and_precrystalize(arraylist* particles, arraylist* precrystalize, struct space* space, int iterazioni){

    for (int i = particles->used-1 ; i >= 0; i--){  //itero su tutto il vettore delle particelle in movimento
                                                    //partendo dalla fine cosi è piu semplice aggiornare il vettore 
        particle p = particles->array[i];

        if(check_crystal_neighbor(space,  &p)){     //se la particella ha cristalli nelle vicinanze (adiacenti)-> si deve precristallizzare
            insertArray(precrystalize,  &p);        //inserisco la particella nel vettore dei precristalli
            removeAt(particles, i);                 //rimuovo la particella da cristallizare dal vettore delle particelle in movimento
        }
        else{       //se la particella NON ha cristalli nelle vicinanze (adiacenti)-> si deve muovere
            int x_movement;
            int y_movement;

            int x = (lcg64_temper_p(&p) % 3) - 1;
            int y = (lcg64_temper_p(&p) % 3) - 1;

            x_movement =  p.x + x; // pick random x direction
            y_movement =  p.y + y; // pick random y direction
            if(!is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)){
                x_movement = p.x;
                y_movement = p.y;
            }
            p.x = x_movement;
            p.y = y_movement;

            particles->array[i] = p;        //assegno alla particella la nuova posizione in cui si è mossa
        }
    }
}
//cristallizzo (aggiorno la matrice)
void crystallize(arraylist* precrystalize, struct space* space){
    
    for(int i = 0; i < precrystalize->used; i++){
        space->field[precrystalize->array[i].x][precrystalize->array[i].y] = 1; //cristallizzo tutti le particelle che si sono precristallizate
    }
    precrystalize->used=0;      //setto used a 0 in modo che mi va a sovrascrivere i dati

}

//creo vettore delle particelle in movimento
void build_vector_particle(arraylist* particles_final, int numero_particelle, int len_x, int len_y, int posizione_seed_x, int posizione_seed_y){
    particle seed = {posizione_seed_x, posizione_seed_y};

    for (int i = 0; i < numero_particelle; i++){
        int rng_seed = (7 + i) * (7 * i + 1);
        particle p;
        do
        {
            //randomicamente scelgo la posizione iniziale della particella
            p.x = lcg64_temper_i(rng_seed++) % len_x;
            p.y = lcg64_temper_i(rng_seed++) % len_y;
        }while(p.x == seed.x && p.y == seed.y);
        p.rng = lcg64_temper_i(rng_seed);
        insertArray(particles_final, &p);       //inserisce la particella generata randomicamente nel vettore
    }
}


int start_crystalline_growth(const int x, const int y, const int iterazioni, const int numero_particelle,
           const int posizione_seed_x, const int posizione_seed_y, int write_out){

    struct space space;     //contiene la matrice
    space.len_x=x;
    space.len_y=y;

    arraylist particles, precrystallized_particles;

    initArray(&particles, numero_particelle + 1);    //inizializzo la lista delle particelle

    initArray(&precrystallized_particles, 10);      //iniziallizzo la lista precristalli


   
   
    
    build_field(&space);         //costruisco il campo
    init_field(&space, posizione_seed_x, posizione_seed_y);      //inizializzo campo
    build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, posizione_seed_x, posizione_seed_y); //costruisco vettore delle particelle iniziali
    
    for(int k = 0; k < iterazioni && particles.used>0; k++){       //itero finchè ho particelle o finchè non finiscono le iterazioni
        move_and_precrystalize(&particles, &precrystallized_particles, &space, iterazioni);       //muovo e precristallizzo
        crystallize(&precrystallized_particles, &space);        //cristallizzo i precristall (aggiorno la matrice)
    }

    return write_out == 1? write_output(&space): 0;     //stampa se write_out==1
}