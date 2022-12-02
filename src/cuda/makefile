CC = nvcc
CFLAGS = -O3 -use_fast_math

schrodinger.x: schrodinger.cu 
	$(CC) -o schrodinger.x schrodinger.cu $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.x *.o
