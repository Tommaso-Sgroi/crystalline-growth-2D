#include <stdlib.h>

typedef struct {

    size_t particles; // particelle che si tovano nella cella e si sono mosse O non possono muoversi
    size_t particles_moved_in; //particelle mosse in una cella
    short status; // -1 non è un cristallo // 0 precristallizzazione // 1 cristallo 
    
    size_t x;
    size_t y;

}cell;

