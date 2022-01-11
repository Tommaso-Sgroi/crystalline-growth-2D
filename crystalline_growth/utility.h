#include <stdlib.h>
#include "../datastructures/dynamiclist.h"

#define RAND_SIZE_T() ({size_t retval;\
                          retval = 0;\
                        for (int i=0; i<64; i++) {\
                            retval = retval*2 + rand()%2;\
                        }\
                        retval;\
                    })


int write_output(struct space* s){
    FILE *f = fopen("output.space", "w");
    if (f == NULL){
        printf("Error opening file!\n");
        exit(1);
    }


    for(int x = 0; x < s->len_x; x++){
        for(int y = 0; y < s->len_y; y++){
            fprintf(f, "%s", s->field[x][y] == 1? "C": "0");
        }
        fprintf(f, "%s", "\n");
    }
    fclose(f);

    return 0;
}