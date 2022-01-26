//#include "../datastructures/dynamiclist.h"
#include "utility.h"
#include <mpi.h>
#include <stddef.h>
#include <string.h>

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


void collect__precrystallized_particles_number(int precrystallized_particles_number, int* vet_precristalxHost){
    MPI_Allgather(&precrystallized_particles_number, 1, MPI_INT, vet_precristalxHost, 1, MPI_INT, MPI_COMM_WORLD);
}

void calculate_allgaterv_offset(int* disp, int* vect_precristal_x_host, int Nhost){
    disp[0] = 0;
    for (int i = 1; i < Nhost; i++){
        disp[i] = disp[i - 1] + vect_precristal_x_host[i - 1];
    }
}

int particles_remaining(int particles_remaining){
    int max_crystallized_particles = 0;
    MPI_Allreduce(&particles_remaining, &max_crystallized_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return max_crystallized_particles;
}

int total_particles_crystallized(arraylist* crystallized){
    int max_crystallized_particles = 0;
    MPI_Allreduce(&crystallized->used, &max_crystallized_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return max_crystallized_particles;
}


void move(arraylist* particles, arraylist* precrystalize, struct space* space){
    
    for(int i = particles->used - 1; i + 1 > 0; i--){
        particle p = particles->array[i];
        
        if(check_crystal_neighbor(space,  &p)){
            insertArray(precrystalize,  &p);
            removeAt(particles, i);
        }
        else{
            int x_movement;
            int y_movement;

            //printf("Muovo particella n\n");
            do{
                int x = 0;
                int y = 0;
                x = (lcg64_temper_p(&p) % 3) - 1;
                y = (lcg64_temper_p(&p) % 3) - 1;

                x_movement =  p.x + x; // pick random x direction
                y_movement =  p.y + y; // pick random y direction
                //printf("(%i, %i) -> (%i, %i), seed: %i\n", p.x, p.y, x_movement, y_movement, p.rng);                                      // sostituibile con (!_movement | y_movement)

            }while(0 == is_in_bounds(x_movement, y_movement, space->len_x, space->len_y)); // finché non sceglie una direzione corretta continua a scegliere randomicamente
            //printf("(%i, %i) -> (%i, %i), seed: %i\n", p.x, p.y, x_movement, y_movement, p.rng);                                      // sostituibile con (!_movement | y_movement)
            p.x = x_movement; 
            p.y = y_movement;
            
            p.rng = lcg64_temper_i(p.rng);
            particles->array[i] = p;

        }

    }
}


void crystalize(particle* buffer_particle_precrystallized, struct space* space, int total_crystallized){
    for(size_t i = 0; i < total_crystallized; i++){
        if(buffer_particle_precrystallized[i].x >= 0){ 
            particle p = buffer_particle_precrystallized[i];
            space->field[p.x][p.y] = 1;
            buffer_particle_precrystallized[i].x = -1;
        }
    }
}

void exchange_particles(arraylist* precrystalize, particle* buffer_particle_precrystallized, int* vet_precristal_x_host, int* disp,
                         MPI_Datatype mpi_particle_type){
    
    MPI_Allgatherv(precrystalize->array, precrystalize->used, mpi_particle_type,
                        buffer_particle_precrystallized, vet_precristal_x_host, disp, mpi_particle_type, MPI_COMM_WORLD);

}

void move_and_crystalize(arraylist* particles, arraylist* precrystalize, 
                            struct space* space, int iterazioni, MPI_Datatype* mpi_particle_type,
                            int Nhost, int rank){
    

    int vec_precristallized_for_host [Nhost];
    int displacement [Nhost];

    particle buffer_particle_precrystallized[precrystalize->size];
    memset(buffer_particle_precrystallized, -1, precrystalize->size);

    for(int k = 0; k < iterazioni; k++){
        
        //printf("RANK: %i; i: %i\n", rank, k);
        move(particles, precrystalize, space);

        collect__precrystallized_particles_number(precrystalize->used, vec_precristallized_for_host);

        calculate_allgaterv_offset(displacement, vec_precristallized_for_host, Nhost);

        exchange_particles(precrystalize, buffer_particle_precrystallized, vec_precristallized_for_host, displacement, *mpi_particle_type);
        
        int total_crystallized = total_particles_crystallized(precrystalize);

        crystalize(buffer_particle_precrystallized, space, total_crystallized);

        if(particles_remaining(particles->used) == 0) break;
        precrystalize->used = 0;
    }
}  


void build_vector_particle(arraylist* particles_final, int numero_particelle, int len_x, int len_y, particle seed){
    
    for (int i = 0; i < numero_particelle; i++){
        int rng_seed = (7 + i) * (7 * i + 1);
        printf("Seed: %i\n", rng_seed);
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
           particle seed, int write_out){
    MPI_Datatype mpi_particle_type;
    int IDhost, Nhost;
    MPI_Comm_rank(MPI_COMM_WORLD, &IDhost);
    MPI_Comm_size(MPI_COMM_WORLD, &Nhost);


    struct space space;
    space.len_x = x;
    space.len_y = y;
    

    arraylist particles, local_particles, precrystallized_particles/*,local_precristal*/;

    int local_n = numero_particelle / Nhost + numero_particelle % Nhost;
    
    build_mpi_particle(&mpi_particle_type);
    build_field(&space); //costruisco il campo
    init_field(&space, seed); //inizializzo campo

    //vettore particelle locali (su ogni host)
    initArray(&local_particles, local_n);
    initArray(&precrystallized_particles, local_n * Nhost);
    if (IDhost == 0) {
        //print_field(&space);
        initArray(&particles, local_n * Nhost);

        build_vector_particle(&particles, numero_particelle, space.len_x, space.len_y, seed); //solo host 0 genera il vettore con tutte le particelle
        print_array(&particles);
    }
    //distribuisce il vettore delle particelle su tutti gli host in local particles
    MPI_Scatter(particles.array, local_n, mpi_particle_type, local_particles.array, local_n, mpi_particle_type, 0, MPI_COMM_WORLD);
    freeArray(&particles);

    //calcolo la size degli array
    local_particles.used = calculate_used_size(&local_particles);
    move_and_crystalize(&local_particles, &precrystallized_particles, &space, iterazioni, &mpi_particle_type, Nhost, IDhost);

    MPI_Type_free(&mpi_particle_type);
    if (IDhost == 0 && write_out == 1){
        return write_output(&space);
    }

    free_field(&space);
    return 0;
}
