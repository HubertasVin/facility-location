build:
	gcc -fopenmp -lstdc++ -O2 rl.cpp -lm -o rl

run:
	./rl

build_run: build run
