#include "../datastructures/dynamiclist.c"

struct {
        cell** field;
        size_t len_x, len_y;
} plan;

cell** field;

arraylist particles_movement, precrystalized_particles;

static size_t len_x, len_y;

void print_field(){

    for(size_t i = 0; i < len_x; i++){
        for(size_t j = 0; j < len_y; j++){
                char* c;
                if(field[i][j].status == -1){
                        c = " P";
                }
                if(field[i][j].status == 0){
                        c = "PC";
                }
                if(field[i][j].status == 1){
                        c = " C";
                }
                printf("%s%zu ", c, field[i][j].particles);
        }
        printf("\n");
    }
        
}

void print_cell(cell* c){
        printf("Particelle: %zu\nParticelle mosse dentro: %zu\nStatus: %i\nCoordinate: (%zu,%zu)\n", 
               c->particles, c->particles_moved_in, c->status, c->x, c->y);
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
cell** build_field(const size_t x, const size_t y){

        field = (cell**) calloc(x, sizeof(cell*));
        for(size_t i = 0; i < x; i++){
                field[i] = (cell*) calloc(y, sizeof(cell)); // TODO PARALLELIZZARE
        }
        return field;
}

size_t rand_size_t() {
  size_t r = 0;
  for (int i=0; i<64; i++) {
    r = r*2 + rand()%2;
  }
  return r;
}

void init_field(const size_t len_x, const size_t len_y, const size_t posizione_seed_x, const size_t posizione_seed_y, const size_t numero_particelle){ 
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
        
        // creo una array con le celle che si trovano sul perimetro
        // della matrice e genero su di esso casualmente le particelle
        cell* perimeter_cells [len_x * 2 + len_y * 2];
        size_t index = 0;
        for(size_t x = 0; x < len_x; x++){
                for(size_t y = 0; y < len_y; (x > 0 && x < len_x - 1)? y += len_y - 1: y++){
                        perimeter_cells[index++] = &field[x][y];
                }
        }
        //printf("%zu\n", index);
        // aggiunge randomicamente le particelle
        for(size_t particles = 0; particles < numero_particelle; particles++){
                perimeter_cells[rand_size_t() % index]->particles++;
        }

        //aggiunge nella lista delle particelle in movimento le celle in cui sono spawnate particelle
        for(size_t i = 0; i < index; i++){       
                if(perimeter_cells[i]->particles > 0){
                        insertArray(&particles_movement, perimeter_cells[i]);
                }
        }
                        

}



bool check_crystal_neighbor(cell* c){
        static short points []= {-1, -1,
                         -1, 0,
                         -1, 1,
                          0, -1,
                          0, 1,
                          1, -1,
                          1, 0,
                          1, 1};

        for (int i = 0, x = 0; i < 8; i++, x++){
                int dx = points[i];
                int dy = points[++i];

                int new_x = c->x + dx;
                int new_y = c->y + dy;

                if(is_in_bounds(new_x, new_y) && 
                  field[new_x][new_y].status == 1){
                
                        return true;
                }
        }
        return false;
}

void move_and_precrystalize(){

        for(size_t i = 0; i < particles_movement.used; i++){
                
                cell* cella = particles_movement.array[i];

                print_cell(cella);
                printf("\n");

                if(check_crystal_neighbor(cella)){ // se si trova vicino a un cristallo vai in precristallizzazione
                        cella->status = 0;
                        removeElement(&particles_movement, cella);
                        insertArray(&precrystalized_particles, cella);

                        printf("Trovato cristallo vicino!\n");
                        print_cell(cella);
                        printf("\n");
                }
                else{ // altrimenti muovi
                
                printf("Muovo le particelle:\n");
                while((cella->particles /*+ 1*/) > 0)
                {
                        printf("\tMuovo particella n° %zu\n", cella->particles);
                        print_cell(cella);
                        printf("\n");

                        short x_movement, y_movement;
                        do{
                                x_movement = cella->x + (rand()%2 * (rand()%2? 1: -1)); // pick random x direction
                                y_movement = cella->y + (rand()%2 * (rand()%2? 1: -1)); // pick random y direction
                        }while(!is_in_bounds(x_movement, y_movement)); // finché non sceglie una direzione corretta continua a scegliere randomicamente

                        // field[cella->x][cella->y].particles--;
                        cella->x = x_movement; // NON VA BENE
                        cella->y = y_movement;

                        printf("\tScelta casella di movimento: (%zu, %zu)\n", cella->x, cella->y);
                        cella->particles-=1;
                        printf("Decremento particelle, rimanenti: %zu", cella->particles);

                        cell* cell_moved_in = &field[x_movement][y_movement];
                        cell_moved_in->particles_moved_in++;

                        printf("Mosso la cella nella nuova cella: \n");
                        print_cell(cell_moved_in);
                        printf("\n");

                        if(cell_moved_in->status == 0){ // se va in una cella in fase di precristallizzazione
                                printf("La cella si trovava in stato 0, aggiorno lo stato:\n");
                                cell_moved_in->particles += cell_moved_in->particles_moved_in;
                                cell_moved_in->particles_moved_in = 0;
                                print_cell(cell_moved_in);
                                printf("\n");
                        }
                        else if(check_crystal_neighbor(cella)){ // se si trova vicino a un cristallo vai in precristallizzazione

                                printf("E' stato rilevato un cristallo vicino,\naggiorno lo stato della cella portandolo in precristallizzazione:\n");
                                cell_moved_in->status = 0;
                                cell_moved_in->particles += cell_moved_in->particles_moved_in;
                                cell_moved_in->particles_moved_in = 0;

                                print_cell(cell_moved_in);
                                printf("\n");

                                removeElement(&particles_movement, cella); // non va bene
                                insertArray(&precrystalized_particles, cella); // non va bene
                        }
                        else{ // se si trova un una casella normale
                             // aggiunge la cella in cui si è mosso nella lista delle celle in cui ci sono particelle da muovere
                             // nel caso il numero di particelle sia 0
                             // aggiorno il numero di particelle spostando le particles_moved_in in particles
                        }
                }
            }
        }
        
        //rimuovo tutte le celle che non contengono più particelle
        printf("rimuovo tutte le celle che non contengono più particelle:\n");
        for(size_t i = 0; i < particles_movement.used; i++){
                cell* c = particles_movement.array[i];
                if(c->particles == 0 &&
                        c->particles_moved_in == 0) // se non ci sono più particelle ne mosse dentro ne stabili
                {
                        printf("RIMOSSO:\n");
                        print_cell(c);
                        printf("\n");
                        removeElement(&particles_movement, c);
                }
                else{
                        printf("AGGIORNO:\n");
                        print_cell(c);
                        c->particles = c->particles_moved_in;
                        print_cell(c);
                        printf("\n");
                }
        }


        // cristallizzo tutte le celle in precristallizzazione
        printf("Cristallizzo le particelle rimanenti:\n");
        for(size_t i = 0; i < precrystalized_particles.used; i++){

                cell* c = precrystalized_particles.array[i];
                printf("AGGIORNO:\n");
                print_cell(c);
                c->status = 1;
                removeElement(&precrystalized_particles, c);
                print_cell(c);
                printf("\n");
        }
}

int start_crystalline_growth(const size_t x, const size_t y, const size_t iterazioni, const size_t numero_particelle,
           const size_t posizione_seed_x, const size_t posizione_seed_y){

        len_x = x;
        len_y = y;
       
        plan.len_x = x;
        plan.len_y = y;

        printf("x: %zu\ny: %zu\nIterazioni: %zu\nNumero particelle: %zu\nPosizione seed: (%zu, %zu)\n",
            x,      y,      iterazioni,      numero_particelle,      posizione_seed_x, posizione_seed_y);
        
        //inizializzo di dynamic
        initArray(&particles_movement, x * 2 + y * 2);
        initArray(&precrystalized_particles, x * 2 + y * 2);
        
        //costruisco il campo e lo inizializzo
        plan.field = build_field(x, y);
        init_field(x, y, posizione_seed_x, posizione_seed_y, numero_particelle);
        
        print_field();
        move_and_precrystalize();
        printf("\n\n");
        print_field();

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