#include "../datastructures/dynamiclist.h"




// arraylist particles_movement, precrystalized_particles;

// static size_t len_x, len_y;


struct space space;

arraylist particles, precrystallized_particles;

void move_and_precrystalize(){

        // for(size_t i = 0; i < particles_movement.used; i++){
                
        //         cell* cella = particles_movement.array[i];

        //         print_cell(cella);
        //         printf("\n");

        //         if(check_crystal_neighbor(cella)){ // se si trova vicino a un cristallo vai in precristallizzazione
        //                 cella->status = 0;
        //                 removeElement(&particles_movement, cella);
        //                 insertArray(&precrystalized_particles, cella);

        //                 printf("Trovato cristallo vicino!\n");
        //                 print_cell(cella);
        //                 printf("\n");
        //         }
        //         else{ // altrimenti muovi
                
        //         printf("Muovo le particelle:\n");
        //         while((cella->particles /*+ 1*/) > 0)
        //         {
        //                 printf("\tMuovo particella n° %zu\n", cella->particles);
        //                 print_cell(cella);
        //                 printf("\n");

        //                 short x_movement, y_movement;
        //                 do{
        //                         x_movement = cella->x + (rand()%2 * (rand()%2? 1: -1)); // pick random x direction
        //                         y_movement = cella->y + (rand()%2 * (rand()%2? 1: -1)); // pick random y direction
        //                 }while(!is_in_bounds(x_movement, y_movement)); // finché non sceglie una direzione corretta continua a scegliere randomicamente

        //                 // field[cella->x][cella->y].particles--;
        //                 cella->x = x_movement; // NON VA BENE
        //                 cella->y = y_movement;

        //                 printf("\tScelta casella di movimento: (%zu, %zu)\n", cella->x, cella->y);
        //                 cella->particles-=1;
        //                 printf("Decremento particelle, rimanenti: %zu", cella->particles);

        //                 cell* cell_moved_in = &field[x_movement][y_movement];
        //                 cell_moved_in->particles_moved_in++;

        //                 printf("Mosso la cella nella nuova cella: \n");
        //                 print_cell(cell_moved_in);
        //                 printf("\n");

        //                 if(cell_moved_in->status == 0){ // se va in una cella in fase di precristallizzazione
        //                         printf("La cella si trovava in stato 0, aggiorno lo stato:\n");
        //                         cell_moved_in->particles += cell_moved_in->particles_moved_in;
        //                         cell_moved_in->particles_moved_in = 0;
        //                         print_cell(cell_moved_in);
        //                         printf("\n");
        //                 }
        //                 else if(check_crystal_neighbor(cella)){ // se si trova vicino a un cristallo vai in precristallizzazione

        //                         printf("E' stato rilevato un cristallo vicino,\naggiorno lo stato della cella portandolo in precristallizzazione:\n");
        //                         cell_moved_in->status = 0;
        //                         cell_moved_in->particles += cell_moved_in->particles_moved_in;
        //                         cell_moved_in->particles_moved_in = 0;

        //                         print_cell(cell_moved_in);
        //                         printf("\n");

        //                         removeElement(&particles_movement, cella); // non va bene
        //                         insertArray(&precrystalized_particles, cella); // non va bene
        //                 }
        //                 else{ // se si trova un una casella normale
        //                      // aggiunge la cella in cui si è mosso nella lista delle celle in cui ci sono particelle da muovere
        //                      // nel caso il numero di particelle sia 0
        //                      // aggiorno il numero di particelle spostando le particles_moved_in in particles
        //                 }
        //         }
        //     }
        // }

}

int start_crystalline_growth(const size_t x, const size_t y, const size_t iterazioni, const size_t numero_particelle,
           const size_t posizione_seed_x, const size_t posizione_seed_y){

        space.len_x = x;
        space.len_y = y;

        printf("x: %zu\ny: %zu\nIterazioni: %zu\nNumero particelle: %zu\nPosizione seed: (%zu, %zu)\n",
            x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);
        
        //inizializzo la lista delle particelle
        initArray(&particles, numero_particelle + 1);
        initArray(&precrystallized_particles, numero_particelle / 4);
        
        
        //costruisco il campo e lo inizializzo
        space.field = build_field(x, y);
        init_field(&space, x, y, posizione_seed_x, posizione_seed_y);
        
        print_field(&space);
        move_and_precrystalize();
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