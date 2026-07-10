// Standalone Prolix NNUE feature-transformer permuter.
// Build: clang++ -O3 -std=c++17 nnue_permuter.cpp -o nnue_permuter
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using I16 = int16_t;
using U16 = uint16_t;
using U64 = uint64_t;
constexpr int L1 = 1024;
constexpr int HalfL1 = L1 / 2;
constexpr int L2 = 16;
constexpr int Buckets = 11;
constexpr int OutBuckets = 8;
constexpr size_t FeatureBytes = size_t(Buckets) * 1536 * L1;
constexpr size_t BiasOffset = FeatureBytes;
constexpr size_t PSQBytes = FeatureBytes + L1 * 2;
constexpr size_t Layer2Bytes = size_t(OutBuckets) * L2 * (L1 + 4);
constexpr size_t NetBytes = PSQBytes + Layer2Bytes + size_t(OutBuckets) * 132;
// clang-format off
constexpr std::array<int, 64> KingBuckets = {
   0,  2,  4,  6,  7,  5,  3,  1,
   8,  8, 10, 10, 11, 11,  9,  9,
  12, 12, 14, 14, 15, 15, 13, 13,
  16, 16, 18, 18, 19, 19, 17, 17,
  16, 16, 18, 18, 19, 19, 17, 17,
  20, 20, 20, 20, 21, 21, 21, 21,
  20, 20, 20, 20, 21, 21, 21, 21,
  20, 20, 20, 20, 21, 21, 21, 21
};
constexpr std::array<int, 12> InverseConvert = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11};
// clang-format on

struct Piece {
  int color, type, square;
};
struct Position {
  std::vector<Piece> pieces;
  int king[2] = {-1, -1};
};
struct Candidate {
  int gain, first, second;
};

static I16 add_wrap(I16 a, I16 b) { return I16(U16(a) + U16(b)); }
static U64 pop_intersection(const U64 *a, const U64 *b, size_t n) {
  U64 result = 0;
  for (size_t i = 0; i < n; i++)
    result += __builtin_popcountll(a[i] & b[i]);
  return result;
}

static Position parse_fen(const std::string &line) {
  Position p;
  int rank = 7, file = 0;
  for (char c : line) {
    if (c == ' ') {
      break;
    }
    if (c == '/') {
      rank--;
      file = 0;
      continue;
    }
    if (c >= '1' && c <= '8') {
      file += c - '0';
      continue;
    }
    int type = c == 'P' || c == 'p'                           ? 0
               : c == 'A' || c == 'a' || c == 'B' || c == 'b' ? 1
               : c == 'F' || c == 'f' || c == 'Q' || c == 'q' ? 2
               : c == 'N' || c == 'n'                         ? 3
               : c == 'R' || c == 'r'                         ? 4
                                                              : 5;
    int color = c >= 'A' && c <= 'Z' ? 0 : 1;
    int square = rank * 8 + file++;
    p.pieces.push_back({color, type, square});
    if (type == 5) {
      p.king[color] = square;
    }
  }
  if (p.king[0] < 0 || p.king[1] < 0) {
    throw std::runtime_error("FEN is missing a king");
  }
  return p;
}

class Net {
public:
  explicit Net(const std::string &path) {
    std::ifstream input = open(path);
    bytes.assign(std::istreambuf_iterator<char>(input), {});
    if (bytes.size() < NetBytes) {
      throw std::runtime_error(
          "NNUE file is too small for the Prolix 1024x16 network");
    }
  }
  std::array<U64, 8> zeroes(const Position &p, int color) const {
    std::array<I16, L1> acc{};
    std::memcpy(acc.data(), bytes.data() + BiasOffset, sizeof(acc));
    int bucket = KingBuckets[(56 * color) ^ p.king[color]],
        raw_bucket = bucket / 2;
    for (const Piece &piece : p.pieces) {
      int pp = (color ^ piece.color) * 6 + piece.type;
      int square = (56 * color) ^ piece.square ^ (bucket & 1 ? 7 : 0);
      int feature = 64 * pp + square;
      size_t vector = size_t(raw_bucket) * 768 +
                      InverseConvert[feature / 64] * 64 + feature % 64;
      std::array<I16, L1> weights{};
      std::memcpy(weights.data(), bytes.data() + vector * L1 * sizeof(I16),
                  sizeof(weights));
      for (int i = 0; i < L1; i++) {
        acc[i] = add_wrap(acc[i], weights[i]);
      }
    }
    std::array<U64, 8> result{};
    for (int i = 0; i < HalfL1; i++) {
      int a = std::clamp<int>(acc[i], 0, 255),
          b = std::min<int>(acc[i + HalfL1], 255);
      int activation = std::clamp((a * b) >> 9, 0, 255);
      if (!activation) {
        result[i / 64] |= U64(1) << (i % 64);
      }
    }
    return result;
  }
  std::vector<char> bytes;

private:
  static std::ifstream open(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    return in;
  }
};

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: nnue_permuter positions.epd input.nnue output.nnue\n";
    return 1;
  }
  try {
    Net net(argv[2]);
    std::ifstream epd(argv[1]);
    if (!epd) {
      throw std::runtime_error("cannot open EPD file");
    }
    std::vector<std::array<U64, 8>> masks;
    std::string line;
    while (std::getline(epd, line)) {
      if (!line.empty()) {
        Position p = parse_fen(line);
        masks.push_back(net.zeroes(p, 0));
        masks.push_back(net.zeroes(p, 1));
      }
    }
    const size_t samples = masks.size(), words = (samples + 63) / 64;
    std::vector<U64> zeroes(HalfL1 * words);
    for (size_t s = 0; s < samples; s++) {
      for (int w = 0; w < 8; w++) {
        for (U64 bits = masks[s][w]; bits; bits &= bits - 1) {
          int n = 64 * w + __builtin_ctzll(bits);
          zeroes[n * words + s / 64] |= U64(1) << (s % 64);
        }
      }
    }
    std::array<int, HalfL1> perm{};
    std::iota(perm.begin(), perm.end(), 0);
    auto z = [&](int i) { return &zeroes[perm[i] * words]; };
    U64 swaps = 0;
    for (int pass = 0; pass < 50; pass++) {
      std::vector<U64> rest(HalfL1 * words);
      std::array<U64, HalfL1> current{};
      for (int i = 0; i < HalfL1; i++) {
        U64 *r = &rest[i * words];
        std::fill(r, r + words, ~U64(0));
        for (int j = i & ~3; j < (i & ~3) + 4; j++) {
          if (j != i) {
            for (size_t w = 0; w < words; w++) {
              r[w] &= z(j)[w];
            }
          }
        }
        current[i] = pop_intersection(z(i), r, words);
      }
      std::vector<Candidate> candidates;
      for (int i = 0; i < HalfL1; i++) {
        for (int j = i + 1; j < HalfL1; j++) {
          if (i / 4 != j / 4) {
            int gain = int(pop_intersection(z(j), &rest[i * words], words) +
                           pop_intersection(z(i), &rest[j * words], words) -
                           current[i] - current[j]);
            if (gain > 0) {
              candidates.push_back({gain, i, j});
            }
          }
        }
      }
      std::sort(candidates.begin(), candidates.end(),
                [](auto a, auto b) { return a.gain > b.gain; });
      std::array<bool, HalfL1 / 4> used{};
      int count = 0;
      for (auto c : candidates) {
        if (!used[c.first / 4] && !used[c.second / 4]) {
          used[c.first / 4] = used[c.second / 4] = true;
          std::swap(perm[c.first], perm[c.second]);
          count++;
        }
      }
      swaps += count;
      if (!count) {
        break;
      }
    }
    auto permute_vector = [&](char *v) {
      std::array<I16, L1> old{}, out{};
      std::memcpy(old.data(), v, sizeof(old));
      for (int i = 0; i < HalfL1; i++) {
        out[i] = old[perm[i]];
        out[HalfL1 + i] = old[HalfL1 + perm[i]];
      }
      std::memcpy(v, out.data(), sizeof(out));
    };
    for (size_t i = 0; i < size_t(Buckets) * 768; i++) {
      permute_vector(net.bytes.data() + i * L1 * sizeof(I16));
    }
    permute_vector(net.bytes.data() + BiasOffset);
    for (int b = 0; b < OutBuckets; b++) {
      for (int o = 0; o < L2; o++) {
        char *row = net.bytes.data() + PSQBytes + (b * L2 + o) * L1;
        std::array<char, L1> old{};
        std::memcpy(old.data(), row, L1);
        for (int i = 0; i < HalfL1; i++) {
          row[i] = old[perm[i]];
          row[HalfL1 + i] = old[HalfL1 + perm[i]];
        }
      }
    }
    std::ofstream out(argv[3], std::ios::binary);
    out.write(net.bytes.data(), net.bytes.size());
    std::ofstream map(std::string(argv[3]) + ".perm");
    for (int i = 0; i < HalfL1; i++) {
      map << i << ' ' << perm[i] << '\n';
    }
    std::cout << "Wrote " << argv[3] << " after " << swaps << " swaps from "
              << samples / 2 << " positions\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
