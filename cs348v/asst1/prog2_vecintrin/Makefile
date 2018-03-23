all: myexp

logger.o: logger.cpp logger.h CS348Vintrin.h CS348Vintrin.cpp
	g++ -c logger.cpp

CS348Vintrin.o: CS348Vintrin.cpp CS348Vintrin.h logger.cpp logger.h
	g++ -c CS348Vintrin.cpp

myexp: CS348Vintrin.o logger.o main.cpp
	g++ -I../common logger.o CS348Vintrin.o main.cpp -o myexp

clean:
	rm -f *.o myexp
