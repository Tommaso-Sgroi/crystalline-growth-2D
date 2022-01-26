#include "crystalline_growth/crystalline_growth.c"


int string_to_int(const char*);

int main(int argc, const char* argv[]) {

    int x = string_to_int(argv[1]);
    int y = string_to_int(argv[2]);
    int iterazioni = string_to_int(argv[3]);
    int numero_particelle = string_to_int(argv[4]);
    int posizione_seed_x = string_to_int(argv[5]);
    int posizione_seed_y = string_to_int(argv[6]);
    int output = string_to_int(argv[7]);


    particle seed = {posizione_seed_x, posizione_seed_y};

    int IDhost, Nhost;

    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &IDhost);
    MPI_Comm_size(MPI_COMM_WORLD, &Nhost); 

    int out = start_crystalline_growth(x, y, iterazioni, numero_particelle, seed, IDhost, Nhost, output);
    MPI_Finalize();
    return out;
}


int string_to_int(const char* c){
    int len = 0;
    if (1 == sscanf(c, "%i", &len)){
        return len;
    }
    return -1;
}