#include "crystalline_growth/crystallinie_growth.c"



int string_to_int(const char*);

int main(int argc, const char* argv[]) {

    int x = string_to_int(argv[1]);
    int y = string_to_int(argv[2]);
    int iterazioni = string_to_int(argv[3]);
    int numero_particelle = string_to_int(argv[4]);
    int posizione_seed_x = string_to_int(argv[5]);
    int posizione_seed_y = string_to_int(argv[6]);

    // arraylist a;
    // initArray(&a, 10);

    // insertArray(&a, 1);
    // insertArray(&a, 2);
    // insertArray(&a, 3);
    // insertArray(&a, 4);
    // insertArray(&a, 5);
 

    return start_crystalline_growth(x, y, iterazioni, numero_particelle, posizione_seed_x, posizione_seed_y);
}


int string_to_int(const char* c){
    int len = 0;
    if (1 == sscanf(c, "%i", &len)){
        return len;
    }
    return -1;
}