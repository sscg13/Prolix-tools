#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
	if (argc < 6) {
		std::cerr << "Proper usage: (exe command) filepath #l1 #inputbuckets #outputbuckets name";
		return 0;
	}
	std::ifstream nnueweights;
	nnueweights.open(argv[1], std::ifstream::binary);
	int l1 = atoi(argv[2]);
	int ib = atoi(argv[3]);
	int ob = atoi(argv[4]);
	std::string name = argv[5];
	int len = name.length();
	if (len > 48) {
		len = 48;
	}
	char blank[1024] = {'\0'};
	char header[16] = {'C', 'B', 'N', 'F', (char)1, '\0', '\0', '\0', '\0', (char)1, (char)1, (char)(l1 % 256), (char)(l1 / 256), (char)ib, (char)ob, (char)len};
	std::ofstream nnueconvert;
	nnueconvert.open(name+".nnue", std::ofstream::binary);
	nnueconvert.write(header, 16);
	nnueconvert.write(argv[5], len);
	if (len < 48) {
		nnueconvert.write(blank, 48-len);
	}
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
	std::cout << "conversion success\n";
	return 0;
}
