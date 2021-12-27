CC=gcc
CFLAGS=-Wall
COUT=-o
CDEBUGFLAG=-g

main: main.c
	$(CC) $(CFLAGS) $(COUT) out/main.o main.c

main_debug: main.c
	$(CC) $(CDEBUGFLAG) $(CFLAGS) $(COUT) out/debug/main.db main.c

.PHONY: clean
clean:
	find out -maxdepth 5 -type f -delete
