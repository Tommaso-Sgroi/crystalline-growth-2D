#include "space.h"

/*struttura arraylist contiene un vettore di unsigned long che rappresentano il valore numerico della locazione in memoria dell'oggetto
oppure di contenere un valore numerico positivo (come il client fd)*/
typedef struct {
  particle *array; //vettore di unsigned long
  size_t used; //lunghezza usata
  size_t size; //lunghezza dell'array totale

  // pthread_mutex_t mutex; //mutex per evitare che i thread accedano alla stessa struttura nello stesso momento
} arraylist;
/*Inizializza l'arraylist con la lunghezza dell'array inziale passata in input,
la mutex non è inizializzata perché potrebbe non servire sempre, bisogna inizializzarla
solo di volontà propria*/
void initArray(arraylist *a, size_t initialSize) {
  a->array = calloc(initialSize, sizeof(particle)); //alloca il vettore
  a->used = 0; //setta a 0 l'utilizzo della lista
  a->size = initialSize; //inserisce il contatore all'inizial size della lista
}

/*inserisce elemento in coda alla lista*/
void insertArray(arraylist *a, particle* element) {
  if(a->used == a->size) { //se gli usati sono uguali agli utilizzati allora bisogna riallocare l'array
    a->size *= 2; //la nuova size è uguale al doppio della vecchia
    a->array = realloc(a->array, a->size * sizeof(particle));//realloca l'array
  }
  a->array[a->used++] = *element; //inserisce l'elemento all'untima posizione e la incrementa
}

/*inserisce l'elemento a una posizione specifica facendo swappare di una posizione in avanti gli altri elementi*/
int insertAt(arraylist* a, size_t where, particle* element){
  if(where < 0 || where >= a->used) return 0; //controlla se la posizione è valida

  if(a->used == a->size) { //incrementa la size se è finita (vedere sopra)
    a->size *= 2;
    a->array = realloc(a->array, a->size * sizeof(particle));
  }

  for(size_t i = a->used; i >= where && i > 0; i--)
    a->array[i] = a->array[i-1]; //mette una posizione avanti tutti gli elementi che si trovano dopo where

  a->array[where] = *element; //mette alla posizione indicata l'elemento
  a->used++; //incrementa gli usati

  return 1; //ritorna 1 su successo
}

/*libera l'array e reimposta le size*/
void freeArray(arraylist *a) {
  free(a->array); //libera il vettore
  a->array = NULL; //nullifica il vecchio puntatore al vettore
  a->used = a->size = 0; //setta a zero i vecchi contatori
}

/*libera e reinizializza l'arraylist a 10*/
void trim_list(arraylist* a){
  freeArray(a);
  initArray(a, 10);
}

/*scambia l'elemento all'indidce i con l'ultimo elemento*/
void switch_last(arraylist* a, size_t i){
  a->array[i] = a->array[a->used-1];
}

//rimuove l'elemento a una posizione specifica scambiando con il primo elemento e cambiando il puntatore 
void removeAt(arraylist* a, size_t where){
  if(where < 0 || where >= a->used) return; //controlla se la posizione è valida

   switch_last(a, where); //fai uno switch dell'ultimo elemento con quello corrente
   // a->array[a->used-1] = NULL; //metti a 0 l'ultimo elemento recentemente puntato
   a->used--; //decrementa used
   // a->array++; // ???
}


void print_array(arraylist* a){
  for(size_t i = 0; i < a->used; i++){
    printf("Coordinates: (%i, %i)\n", a->array[i].x, a->array[i].y);
    // printf("%i\n", a->array[i]);
  }
}

// int fakemain(int argc, char const *argv[])
// {
//   arraylist a;
//   int i;

//   initArray(&a, 5);  // initially 5 elements
//   for (i = 0; i < 10; i++)
//     insertArray(&a, i);  // automatically resizes as necessary
//   printf("usata: %ld\n", a.used);  // print number of elements

//   // removeElement(&a, 1);
//   // removeElement(&a, 5);
//   // removeElement(&a, 9);

//   insertAt(&a, 0, 100);

//   printf("usata dopo aggiunta: %ld\n", a.used);  // print number of elements

//   for (i = 0; i < a.used; i++)
//     printf("%d\n", a.array[i]);
//   printf("size totale: %lu\n", a.size);

//   removeAt(&a, 0);
//    for (i = 0; i < a.used; i++)
//     printf("%d\n", a.array[i]);
//   printf("size : %lu\n", a.used);
//   return 0;
// }

// int main(int argc, char const *argv[])
// {
//   fakemain(argc, argv);
//   return 0;
// }

