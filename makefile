FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<
parallel: nbody.cu compute.cu
	nvcc $(FLAGS) nbody.cu compute.cu -o parallel $(LIBS) 
clean:
	rm -f *.o parallel 
