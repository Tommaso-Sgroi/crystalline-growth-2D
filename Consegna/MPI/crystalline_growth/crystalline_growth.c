//#include "../datastructures/dynamiclist.h"
#include "utility.h"
#include <mpi.h>
#include <stddef.h>
#include <string.h>

void build_mpi_particle(MPI_Datatype* mpi_particle_type){
    // Create the datatype
    int lengths[3] = {1, 1, 1};
 
    MPI_Aint displacements[3];
    particle my_particle;
    MPI_Aint base_address;
    MPI_Get_address(&my_particle, &base_address);
    MPI_Get_address(&my_particle.x, &displacements[0]);
    MPI_Get_address(&my_particle.y, &displacements[1]);
    MPI_Get_address(&my_particle.rng, &displacements[2]);

    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);

 
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(3, lengths, displacements, types, mpi_particle_type);
    MPI_Type_commit(mpi_particle_type);
}


void move_and_precrystalize(arraylist *particles, arraylist *precrystalize,  struct space *space, int IdHost, int iterazione)
{
    for (int i = particles->used-1 ; i >= 0; i--){
        particle p = particles->array[i];

        if (check_crystal_neighbor(space, &p)){ // controlla se vicino ha un cristallo
            insertArray(precrystalize, &p);     //si inserisce nella lista  
            removeAt(particles, i);             //si rimuove dalla lista delle particelle
        }
        else
        {
            int x_movement;
            int y_movement;

            int x = (lcg64_temper_p(&p) % 3) - 1; // sceglie random la direzione x
            int y = (lcg64_temper_p(&p) % 3) - 1; // sceglie random la direzione y

            x_movement =  p.x + x; // pick random x direction
            y_movement =  p.y + y; // pick random y direction
            if(!is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)){
                x_movement = p.x;
                y_movement = p.y;
            }
            p.x = x_movement;
            p.y = y_movement;

            particles->array[i] = p;        //aggiorno posizione della particella 

        }
    }
}

//cristallizzo (aggiorno la matrice su ogni host)
void crystallizes(struct space* space, particle* all_precrystalls, int totale_precristalli){

    for (int i = 0; i < totale_precristalli; i++){
        particle* p = &all_precrystalls[i];
        space->field[p->x][p->y] = 1; // indico che c'è un cristallo
    }
}


void build_vector_particle(arraylist* particles, int numero_particelle, int len_x, int len_y, particle seed){
    
    for (int i = 0; i < numero_particelle; i++){
        int rng_seed = (7 + i) * (7 * i + 1); // sceglie un seed iniziale
        particle p;
        do
        {
            p.x = lcg64_temper_i(rng_seed++) % len_x; // sceglie posizione x 
            p.y = lcg64_temper_i(rng_seed++) % len_y; // sceglie posizione y 
        }while(p.x == seed.x && p.y == seed.y);       // ripete finché non esce una posizione diversa dal seed  
        p.rng = lcg64_temper_i(rng_seed);             //assegna un seed random alla particella
        insertArray(particles, &p);
    }
}    

void reset_array(arraylist *particles){
    
    for(int i = 0; i < particles->used; i++){
        particles->array[i].x = -4;
    }
    particles->used = 0;
}



int start_crystalline_growth(const int x, const int y, const int iterazioni, const int numero_particelle,
           particle seed,int IDhost, int Nhost, int write_out){
    
    MPI_Datatype mpi_particle_type;
    struct space space;
    space.len_x = x;
    space.len_y = y;
    
    int totale_precristalli = 0;                                //precristalli totali cioè di tutte le macchine ( è un count)
    int totale_particelle_rimaste = numero_particelle;
    
    int vet_precristal_x_host [Nhost];  //vettore contenente il numero di precristalli su ogni host
    int vet_particles_x_host[Nhost];    //vettore contenente il numero di particelle su ogni host
    int disp [Nhost];                   //serve per la allgaterv e scatterv, indica dove iniziare a mettere i valori (offset)
    arraylist particles, local_particles, local_precrystallized_particles; //vettore particelle e precristalli locali e totali
    particle all_precrystalls [numero_particelle]; // buffer in cui inserire le particelle scambiate

    //assegna ad ogni host il numero di particelle che deve muovere
    if(numero_particelle%Nhost==0){
        for(int i=0; i<Nhost; i++)
            vet_particles_x_host[i] = numero_particelle / Nhost;
    }
    else{
        for(int i=0; i<Nhost-1; i++)
            vet_particles_x_host[i]=numero_particelle / Nhost;
        vet_particles_x_host[Nhost-1]=numero_particelle-(vet_particles_x_host[0]*(Nhost-1));
    }

    build_mpi_particle(&mpi_particle_type); // costruisce la struttura per scambiare le particelle
    build_field(&space);        //costruisce matrice 
    init_field(&space, seed);

   
    initArray(&local_particles, vet_particles_x_host[IDhost]); //array delle particelle su ogni host
    
    if (IDhost == 0){                                          //solo host zero crea il vettore di tutte le particelle
        initArray(&particles, numero_particelle);
        build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, seed);
    }
    
    calculate_displacement(vet_particles_x_host, disp, Nhost); //offset per scatterv

    //distribuisco a tutti gli host le relative particelle
    MPI_Scatterv(particles.array, vet_particles_x_host, disp, mpi_particle_type, local_particles.array,
                 vet_particles_x_host[IDhost], mpi_particle_type, 0, MPI_COMM_WORLD);
    local_particles.used = vet_particles_x_host[IDhost];

    initArray(&local_precrystallized_particles, vet_particles_x_host[IDhost]); // inizializza il vettore delle particelle precristallizzate
    
    if(IDhost==0)
        freeArray(&particles);

    for (int k = 0; k < iterazioni && totale_particelle_rimaste > 0; k++)
    { 
        totale_particelle_rimaste = 0;

        move_and_precrystalize(&local_particles, &local_precrystallized_particles, &space, IDhost, k);
        
        MPI_Allreduce(&local_particles.used, &totale_particelle_rimaste, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        //ogni host avrà un vettore contenente il numero di particelle precristallizate per ogni altro host
        MPI_Allgather(&local_precrystallized_particles.used, 1, MPI_INT, vet_precristal_x_host, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allreduce(&local_precrystallized_particles.used, &totale_precristalli, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

         
        calculate_displacement(vet_precristal_x_host, disp, Nhost); //offset per allgatherv

        //raccoglie tutti i precristalli (ogni host avrà il vettore di tutti i precristalli creati sui singoli host)
        MPI_Allgatherv(local_precrystallized_particles.array, local_precrystallized_particles.used, mpi_particle_type, 
                       all_precrystalls, vet_precristal_x_host, disp, mpi_particle_type, MPI_COMM_WORLD);

        crystallizes(&space, all_precrystalls, totale_precristalli);

    	reset_array(&local_precrystallized_particles); //svuoto vettore dei precristalli locali
                               
        totale_precristalli = 0;
    }
    freeArray(&local_precrystallized_particles);
    freeArray(&local_particles);
    MPI_Type_free(&mpi_particle_type);

    if(IDhost == 0 && write_out == 1){
        return write_output(&space);        //scrive su file i risultati
    }

    return 0;
}
