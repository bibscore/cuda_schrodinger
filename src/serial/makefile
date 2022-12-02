CC = gcc
CFLAGS = -lm
OBJ = prg.o

ODIR = obj

prg: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
