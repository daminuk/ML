CC = g++
CFLAGS = -std=c++11 -O3
OBJECTS = main.o network.o

main : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o NeuralNetwork

main.o : main.cc
	$(CC) $(CFLAGS) -c main.cc

network.o : network.cc
	$(CC) $(CFLAGS) -c network.cc

clean :
	rm -rf $(OBJECTS) NeuralNetwork
	
