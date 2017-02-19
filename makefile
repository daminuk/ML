CC = g++
CFLAGS = -std=c++11 -O3
OBJECTS = main.o network.o gradient.o evolution.o

main : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o NeuralNetwork

main.o : main.cc
	$(CC) $(CFLAGS) -c main.cc

network.o : network.cc
	$(CC) $(CFLAGS) -c network.cc

evolution.o : evolution.cc
	$(CC) $(CFLAGS) -c evolution.cc

gradient.o : gradient.cc
	$(CC) $(CFLAGS) -c gradient.cc

clean :
	rm -rf $(OBJECTS) NeuralNetwork
