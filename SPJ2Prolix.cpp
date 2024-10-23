#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Proper usage: (exe command) filepath";
		return 0;
	}
	std::ifstream nnueweights;
	nnueweights.open(argv[1], std::ifstream::binary);
	char header[64];
	nnueweights.read(header, 64);
	int l1 = 256*(unsigned char)header[12]+(unsigned char)header[11];
	int ib = header[13];
	int ob = header[14];
	int len = header[15];
	std::string name = "";
	for (int i = 0; i < len; i++) {
		name = name + header[16+i];
	}
	std::ofstream nnueconvert;
	nnueconvert.open(name+".nnue", std::ofstream::binary);
	char *l1weights = new char[1536*l1*ib];
	nnueweights.read(l1weights, 1536*l1*ib);
	int convert[12] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};
	for (int j = 0; j < ib; j++) {
		for (int i = 0; i < 12; i++) {
			nnueconvert.write(l1weights+1536*l1*j+128*l1*convert[i], 128*l1);
		}
	}
	char *remainder = new char[2*l1+4*l1*ob+2*ob];
	nnueweights.read(remainder, 2*l1+4*l1*ob+2*ob);
	nnueconvert.write(remainder, 2*l1+4*l1*ob+2*ob);
	std::cout << "conversion success\n";
	delete[] l1weights;
	delete[] remainder;
	return 0;
}