#include "crystalline_growth/crystallinie_growth.c"



size_t string_to_size_t(const char*);

int main(int argc, const char* argv[]) {

    size_t x = string_to_size_t(argv[1]);
    size_t y = string_to_size_t(argv[2]);
    size_t iterazioni = string_to_size_t(argv[3]);
    size_t numero_particelle = string_to_size_t(argv[4]);
    size_t posizione_seed_x = string_to_size_t(argv[5]);
    size_t posizione_seed_y = string_to_size_t(argv[6]);

    // arraylist a;
    // initArray(&a, 10);

    // insertArray(&a, 1);
    // insertArray(&a, 2);
    // insertArray(&a, 3);
    // insertArray(&a, 4);
    // insertArray(&a, 5);


    return start_crystalline_growth(x, y, iterazioni, numero_particelle, posizione_seed_x, posizione_seed_y);
}


size_t string_to_size_t(const char* c){
    size_t len = 0;
    if (1 == sscanf(c, "%zu", &len)){
        return len;
    }
    return -1;
}