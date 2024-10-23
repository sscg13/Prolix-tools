#include <fstream>
#include <iostream>

int l1 = 64;
int ib = 1;
int ob = 1;
int main(int argc, char* argv[]) {
	if (argc < 5) {
		std::cout << "Proper usage: (exe command) filepath #l1 #inputbuckets #outputbuckets";
		return 0;
	}
	std::ifstream nnueweights;
	nnueweights.open(argv[1], std::ifstream::binary);
	l1 = atoi(argv[2]);
	ib = atoi(argv[3]);
	ob = atoi(argv[4]);
	char blank[1024] = {'\0'};
	std::ofstream nnueconvert;
	nnueconvert.open("converted.nnue", std::ofstream::binary);
	nnueconvert.write(blank, 64);
	char *l1weights = new char[1536*l1*ib];
	nnueweights.read(l1weights, 1536*l1*ib);
	int convert[12] = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11};
	for (int j = 0; j < ib; j++) {
		for (int i = 0; i < 12; i++) {
			nnueconvert.write(l1weights+1536*l1*j+128*l1*convert[i], 128*l1);
		}
	}
	char *remainder = new char[2*l1+4*l1*ob+2*ob];
	nnueweights.read(remainder, 2*l1+4*l1*ob+2*ob);
	nnueconvert.write(remainder, 2*l1+4*l1*ob+2*ob);
	return 0;
}