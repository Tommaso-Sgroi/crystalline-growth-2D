CC=gcc
CFLAGS=-Wall

all: main

main: main.c
	$(CC) $(CFLAGS) -o out/main.o main.c

main_debug: main.c
	$(CC) -g $(CFLAGS) -o out/debug/main.db main.c

.PHONY: clean
clean:
	find out -maxdepth 5 -type f -delete
