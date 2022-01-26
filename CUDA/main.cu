#include "crystalline_growth/crystallinie_growth.cu"



int string_to_int(const char*);
void printDevProp(cudaDeviceProp devProp);

int main(int argc, const char* argv[]) {

    int x = string_to_int(argv[1]);
    int y = string_to_int(argv[2]);
    int iterazioni = string_to_int(argv[3]); 
    int numero_particelle = string_to_int(argv[4]);
    int posizione_seed_x = string_to_int(argv[5]);
    int posizione_seed_y = string_to_int(argv[6]);
    int write_out = string_to_int(argv[7]);


    return start_crystalline_growth(x, y, iterazioni, numero_particelle, posizione_seed_x, posizione_seed_y, write_out);

}

__host__ int string_to_int(const char* c){
    int len = 0;
    if (1 == sscanf(c, "%i", &len)){
        return len;
    }
    return -1;
}
