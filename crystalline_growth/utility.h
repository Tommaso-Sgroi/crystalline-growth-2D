#include <stdlib.h>
#define RAND_SIZE_T() ({size_t retval;\
                          retval = 0;\
                        for (int i=0; i<64; i++) {\
                            retval = retval*2 + rand()%2;\
                        }\
                        retval;\
                    })
