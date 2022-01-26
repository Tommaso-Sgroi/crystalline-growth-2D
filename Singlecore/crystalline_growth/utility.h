#include <stdlib.h>
#include "../datastructures/dynamiclist.h"

#define RAND_SIZE_T() ({size_t retval;\
                          retval = 0;\
                        for (int i=0; i<64; i++) {\
                            retval = retval*2 + rand()%2;\
                        }\
                        retval;\
                    })


int temper(int x)
{
	x ^= x>>11;
	x ^= x<<7 & 0x9D2C5680;
	x ^= x<<15 & 0xEFC60000;
	x ^= x>>18;
	return x;
}

int lcg64_temper_i(int seed){
	seed = 6364136223846793005ULL * seed + 1;
	return temper(seed >> 16);
}

int lcg64_temper_p(particle* seed){
	seed->rng = 6364136223846793005ULL * (seed->rng) + 1;
	return temper(seed->rng >> 16);
}




int write_output(struct space* s){
    FILE *f = fopen("output.space", "w");
    if (f == NULL){
        printf("Error opening file!\n");
        exit(1);
    }


    for(int x = 0; x < s->len_x; x++){
        for(int y = 0; y < s->len_y; y++){
            fprintf(f, "%s ", s->field[x][y] == 1? "C": "0");
        }
        fprintf(f, "%s", "\n");
    }
    fclose(f);

    return 0;
}