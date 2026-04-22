#include <torch/extension.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <string>

#ifdef _MSC_VER
#include <intrin.h> // Required for MSVC intrinsics
inline int get_lsb(uint64_t occ) {
    unsigned long index;
    _BitScanForward64(&index, occ);
    return static_cast<int>(index);
}
#else
inline int get_lsb(uint64_t occ) {
    return __builtin_ctzll(occ);
}
#endif

// 1. Define the 48-byte struct exactly as it exists in the binary file.
// #pragma pack ensures the compiler doesn't add hidden padding bytes.
#pragma pack(push, 1)
struct ChessRecord {
    uint64_t occupancy;
    uint8_t mailbox[32];
    int16_t score_wdl;
    uint8_t stm_king;
    uint8_t nstm_king;
    uint8_t padding[4];
};
#pragma pack(pop)

struct UnpackedBoard {
    uint64_t occupancy;
    uint8_t mailbox[64]; // Fully unpacked (1 piece per index)
    int stm_king_sq;
    int nstm_king_sq;
};

template <int MaxActive>
struct ActiveFeatures {
    int indices[MaxActive];
    float coeffs[MaxActive];
    int count = 0;

    inline void add(int index, float coeff) {
        indices[count] = index;
        coeffs[count] = coeff;
        count++;
    }
    
    inline void clear() { count = 0; }
};

// Fast sigmoid helper
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct MaterialExtractor {
    static constexpr int NUM_FEATURES = 7;
    static constexpr int MAX_ACTIVE = 33; // 32 pieces + 1 bias

    static inline float forward(const UnpackedBoard& board, const float* weights, ActiveFeatures<MAX_ACTIVE>& features) {
        float eval = 0.0f;
        features.add(6, 1.0f); // Bias
        eval += 1.0f * weights[6];
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ); 
            occ &= occ - 1; 
            uint8_t piece = board.mailbox[sq]; 
            int base_piece = piece % 6;
            float coeff = (piece < 6) ? 1.0f : -1.0f;
            features.add(base_piece, coeff);
            eval += coeff * weights[base_piece];
        }
        return eval;
    }
};

struct PRFExtractor {
    static constexpr int NUM_FEATURES = 97;
    static constexpr int MAX_ACTIVE = 65; // 32 pieces * 2 features + 1 bias

    static inline float forward(const UnpackedBoard& board, const float* weights, ActiveFeatures<MAX_ACTIVE>& features) {
        float eval = 0.0f;
        features.add(96, 1.0f); // Bias
        eval += 1.0f * weights[96];
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ); 
            occ &= occ - 1; 
            uint8_t piece = board.mailbox[sq]; 
            int base_piece = piece % 6; 
            
            bool is_stm = (piece < 6);
            float coeff = is_stm ? 1.0f : -1.0f;
            
            int file = sq % 8;
            int rank = sq / 8;
            int relative_rank = is_stm ? rank : (7 - rank); 
            
            int file_idx = (base_piece * 8) + file;
            int rank_idx = 48 + (base_piece * 8) + relative_rank;
            
            features.add(file_idx, coeff);            // File
            eval += coeff * weights[file_idx];
            
            features.add(rank_idx, coeff);            // Rank
            eval += coeff * weights[rank_idx];
        }
        return eval;
    }
};

struct PSTExtractor {
    static constexpr int NUM_FEATURES = 391; // 1 Bias + 6 Material + 384 Full PST
    static constexpr int MAX_ACTIVE = 65;    // Max 32 pieces * 2 features + 1 bias

    static inline float forward(const UnpackedBoard& board, const float* weights, ActiveFeatures<MAX_ACTIVE>& features) {
        float eval = 0.0f;

        // --- 1. Global Bias ---
        features.add(0, 1.0f);
        eval += weights[0];

        // --- 2. Extract Factorized Features ---
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ);
            occ &= occ - 1;

            uint8_t piece = board.mailbox[sq];
            int base_piece = piece % 6;
            bool is_stm = (piece < 6);

            // Perspective flip
            int relative_sq = is_stm ? sq : (sq ^ 56);
            float coeff = is_stm ? 1.0f : -1.0f;

            // Calculate Indices
            int mat_idx = 1 + base_piece;
            int pst_idx = 7 + (base_piece * 64) + relative_sq;

            // Add Base Material
            features.add(mat_idx, coeff);
            eval += coeff * weights[mat_idx];

            // Add PST Delta
            features.add(pst_idx, coeff);
            eval += coeff * weights[pst_idx];
        }

        return eval;
    }
};

struct KPExtractor {
    static constexpr int NUM_FEATURES = 23233;
    static constexpr int MAX_ACTIVE = 129;

    static inline float forward(const UnpackedBoard& board, const float* weights, ActiveFeatures<MAX_ACTIVE>& features) {
        float eval = 0.0f;

        // --- 1. Global Bias (STM Advantage) ---
        features.add(23232, 1.0f);
        eval += weights[23232];

        // --- 2. King Positions ---
        int k_stm_sq = board.stm_king_sq;
        int k_nstm_sq = board.nstm_king_sq; // Actual board square

        // --- 3. Compute Mirrored Bucket Indices (0 to 31) ---
        bool flip_stm = (k_stm_sq % 8) >= 4;
        int k_stm_mirrored = flip_stm ? (k_stm_sq ^ 7) : k_stm_sq;
        int k_idx_stm = ((k_stm_mirrored & 56) >> 1) | (k_stm_mirrored & 3);

        // For NSTM, we flip the board vertically first (relative to their forward)
        int k_nstm_rel = k_nstm_sq ^ 56;
        bool flip_nstm = (k_nstm_rel % 8) >= 4;
        int k_nstm_mirrored = flip_nstm ? (k_nstm_rel ^ 7) : k_nstm_rel;
        int k_idx_nstm = ((k_nstm_mirrored & 56) >> 1) | (k_nstm_mirrored & 3);

        // --- 4. Evaluate All Pieces (Single Pass) ---
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ);
            occ &= occ - 1;

            uint8_t p = board.mailbox[sq];
            int base_piece = p % 6;
            bool is_stm_piece = (p < 6);

            // A. Evaluate from STM's Perspective (Adds to Eval)
            if (sq != k_stm_sq) { // Never evaluate our own King in our own bucket
                // Piece Type: 0-4 (Friendly), 5-9 (Enemy), 10 (Enemy King)
                int pt_stm = (base_piece == 5) ? 10 : (is_stm_piece ? base_piece : base_piece + 5);
                int sq_stm = flip_stm ? (sq ^ 7) : sq;

                int base_idx_stm = pt_stm * 64 + sq_stm;
                int kp_idx_stm = 704 + (k_idx_stm * 11 + pt_stm) * 64 + sq_stm;

                features.add(base_idx_stm, 1.0f);
                features.add(kp_idx_stm, 1.0f);
                eval += weights[base_idx_stm] + weights[kp_idx_stm];
            }

            // B. Evaluate from NSTM's Perspective (Subtracts from Eval)
            if (sq != k_nstm_sq) { // Never evaluate their own King in their own bucket
                // Piece Type: 0-4 (Friendly to them), 5-9 (Enemy to them), 10 (Enemy King to them)
                int pt_nstm = (base_piece == 5) ? 10 : (!is_stm_piece ? base_piece : base_piece + 5);
                int sq_nstm = flip_nstm ? (sq ^ 63) : (sq ^ 56);

                int base_idx_nstm = pt_nstm * 64 + sq_nstm;
                int kp_idx_nstm = 704 + (k_idx_nstm * 11 + pt_nstm) * 64 + sq_nstm;

                features.add(base_idx_nstm, -1.0f);
                features.add(kp_idx_nstm, -1.0f);
                eval -= (weights[base_idx_nstm] + weights[kp_idx_nstm]);
            }
        }

        return eval;
    }
};

// ====================================================================
//                         PP Extractor
// ====================================================================
// Weight for each unordered pair of pieces on the board.
// Features are indexed by (piece_type, square) in [0, 768):
//   piece_type in [0, 12)  -> 0-5 STM, 6-11 NSTM (5 = STM king, 11 = NSTM king)
//   feat = pt * 64 + sq
//   STM features: [0, 384),  NSTM features: [384, 768)
//
// Two symmetries reduce the parameter space to 147072 canonical weights:
//
// (a) Vertical symmetry: w[i][j] = -w[flip(i)][flip(j)]
//       flip(f) = ((pt+6)%12)*64 + (sq^56)  -- swaps colors and mirrors ranks
//     This folds NSTM-NSTM pairs into the STM-STM triangular block (with negation),
//     and halves the STM-NSTM subspace.
//
// (b) Pair symmetry: w[i][j] = w[j][i]  -- unordered pairs, triangular indexing.
//
// Index layout (PP_HALF = 384, PP_SS = PP_HALF*(PP_HALF-1)/2 = 73536):
//   [0,       PP_SS)  : STM-STM pairs, idx = b*(b-1)/2 + a,  0 <= a < b < PP_HALF
//                       NSTM-NSTM pairs map here via flip with sign -1.
//   [PP_SS, 2*PP_SS)  : STM-NSTM pairs, idx = PP_SS + bp*(bp-1)/2 + a,
//                       where a = STM feature, bp = flip(NSTM feature),
//                       canonical: a < bp (swap + negate if a > bp).
//                       Self-symmetric pairs (a == bp) have weight pinned to 0.
//   [2*PP_SS]         : Bias.
//
// Engine indexing uses the same formula with one helper: pp_flip(f).
// ====================================================================
static constexpr int PP_HALF = 6 * 64;                        // 384
static constexpr int PP_SS   = PP_HALF * (PP_HALF - 1) / 2;  // 73536

static inline int pp_flip(int f) {
    int pt = f / 64, sq = f % 64;
    return ((pt + 6) % 12) * 64 + (sq ^ 56);
}

struct PPExtractor {
    static constexpr int NUM_FEATURES = PP_SS * 2 + 1; // 147072 pairs + bias
    static constexpr int MAX_ACTIVE   = 32 * 31 / 2 + 1; // C(32,2) pairs + bias

    static inline float forward(const UnpackedBoard& board, const float* weights,
                                ActiveFeatures<MAX_ACTIVE>& features) {
        const int bias_idx = NUM_FEATURES - 1;
        float eval = weights[bias_idx];
        features.add(bias_idx, 1.0f);

        // Collect piece features (piece_type * 64 + sq) from STM-relative board.
        int feats[32];
        int n = 0;
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ);
            occ &= occ - 1;
            int p = board.mailbox[sq];
            feats[n++] = p * 64 + sq;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int fi = feats[i], fj = feats[j];
                bool si = fi < PP_HALF, sj = fj < PP_HALF;
                float sign = 1.0f;
                int idx;

                if (si == sj) {
                    // Both STM or both NSTM: vertical flip folds NSTM-NSTM into STM-STM.
                    if (!si) { fi = pp_flip(fi); fj = pp_flip(fj); sign = -1.0f; }
                    int a = std::min(fi, fj), b = std::max(fi, fj);
                    idx = b * (b - 1) / 2 + a;
                } else {
                    // Mixed: ensure fi is STM, then reduce via flip of the NSTM feature.
                    if (!si) std::swap(fi, fj);
                    int a  = fi;
                    int bp = pp_flip(fj); // flip(NSTM) -> STM feature in [0, PP_HALF)
                    if (a == bp) continue; // self-symmetric, weight pinned to 0
                    if (a > bp) { std::swap(a, bp); sign = -1.0f; }
                    idx = PP_SS + bp * (bp - 1) / 2 + a;
                }

                features.add(idx, sign);
                eval += sign * weights[idx];
            }
        }
        return eval;
    }
};

// ====================================================================
//                         PPxK Extractor
// ====================================================================
// For each perspective (STM, NSTM), compute 64 intermediate values
// (one per square):
//   I[sq] = sum over ordered piece pairs (a,b), a != b, with sq_a == sq of
//           W[feat_a * 768 + feat_b]
// Equivalently, for each unordered pair {i,j}, W[feat_i * 768 + feat_j]
// flows into I[sq_i], while W[feat_j * 768 + feat_i] flows into I[sq_j].
// Then reduce via king squares:
//   reduced = sum_sq K[king_sq * 64 + sq] * I[sq]
// Final:
//   eval = reduced_stm - reduced_nstm + bias
//
// This is side-to-move agnostic by construction: swapping stm/nstm and
// mirroring the board swaps the two reduced terms, negating the eval.
// No internal symmetry constraint is imposed on W or K.
// ====================================================================
struct PPxKExtractor {
    static constexpr int NUM_PP       = 768 * 768;
    static constexpr int NUM_K        = 64 * 64;
    static constexpr int NUM_FEATURES = NUM_PP + NUM_K + 1;
    // Per perspective: 32*31 ordered-pair W features + 64 K features.
    // Two perspectives + bias.
    static constexpr int MAX_ACTIVE   = 2 * (32 * 31) + 2 * 64 + 1;

    static inline float forward(const UnpackedBoard& board, const float* weights,
                                ActiveFeatures<MAX_ACTIVE>& features) {
        const float* W = weights;                // [768 * 768]
        const float* K = weights + NUM_PP;       // [64 * 64]
        const int bias_idx = NUM_FEATURES - 1;

        float eval = weights[bias_idx];
        features.add(bias_idx, 1.0f);

        // Gather pieces in STM and NSTM perspectives.
        int feats_stm[32], sqs_stm[32];
        int feats_nstm[32], sqs_nstm[32];
        int n = 0;
        uint64_t occ = board.occupancy;
        while (occ) {
            int sq = get_lsb(occ);
            occ &= occ - 1;
            int p = board.mailbox[sq];
            feats_stm[n]  = p * 64 + sq;
            sqs_stm[n]    = sq;
            int p_n  = (p + 6) % 12;
            int sq_n = sq ^ 56;
            feats_nstm[n] = p_n * 64 + sq_n;
            sqs_nstm[n]   = sq_n;
            ++n;
        }

        const int stm_king  = board.stm_king_sq;
        const int nstm_king = board.nstm_king_sq ^ 56; // into NSTM's own view

        float I_stm[64]  = {0.0f};
        float I_nstm[64] = {0.0f};

        // Iterate ordered pairs; each unordered pair is visited twice,
        // once contributing W[i][j] -> I[sq_i], once W[j][i] -> I[sq_j].
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;

                // STM perspective
                int   w_idx_s = feats_stm[i] * 768 + feats_stm[j];
                int   sq_s    = sqs_stm[i];
                float w_s     = W[w_idx_s];
                float k_s     = K[stm_king * 64 + sq_s];
                I_stm[sq_s]  += w_s;
                features.add(w_idx_s, k_s);        // dEval/dW = +K
                eval         += k_s * w_s;

                // NSTM perspective
                int   w_idx_n = feats_nstm[i] * 768 + feats_nstm[j];
                int   sq_n    = sqs_nstm[i];
                float w_n     = W[w_idx_n];
                float k_n     = K[nstm_king * 64 + sq_n];
                I_nstm[sq_n] += w_n;
                features.add(w_idx_n, -k_n);       // dEval/dW = -K
                eval         -= k_n * w_n;
            }
        }

        // K features: gradient wrt K[king][sq] is +-I[sq].
        for (int sq = 0; sq < 64; ++sq) {
            features.add(NUM_PP + stm_king  * 64 + sq,  I_stm[sq]);
            features.add(NUM_PP + nstm_king * 64 + sq, -I_nstm[sq]);
        }

        return eval;
    }
};

// 2. The core processing function
template <typename Extractor>
float process_batch_impl(
    int start_idx, int batch_size, float k, float lam, 
    torch::Tensor weights, torch::Tensor grads, 
    torch::Tensor rawdata, torch::Tensor indices) 
{
    float* w_ptr = weights.data_ptr<float>();
    float* g_ptr = grads.data_ptr<float>();

    const ChessRecord* dataset = reinterpret_cast<const ChessRecord*>(rawdata.data_ptr<uint8_t>());
    const int32_t* permutation = indices.data_ptr<int32_t>();

    float total_loss = 0.0f;

    #pragma omp parallel
    {
        float local_loss = 0.0f;
        std::vector<float> local_grads(Extractor::NUM_FEATURES, 0.0f);

        UnpackedBoard board;
        ActiveFeatures<Extractor::MAX_ACTIVE> features;

        #pragma omp for nowait
        for (int i = 0; i < batch_size; ++i) {
            int actual_idx = permutation[start_idx + i];
            const auto& record = dataset[actual_idx];

            // --- 1. DATA UNPACKING (Done once per position) ---
            board.occupancy = record.occupancy;
            for (int j = 0; j < 32; ++j) {
                board.mailbox[j * 2]     = record.mailbox[j] & 0x0F;
                board.mailbox[j * 2 + 1] = (record.mailbox[j] >> 4) & 0x0F;
            }
            board.stm_king_sq = record.stm_king;
            board.nstm_king_sq = record.nstm_king;

            // --- 2. FEATURE EXTRACTION ---
            features.clear();
            
            // Swap this function call to change your tuner type!
            float eval = Extractor::forward(board, w_ptr, features); 

            // --- 4. LOSS CALCULATION ---
            int16_t packed_val = record.score_wdl;
            int score = packed_val / 3;
            int rem = packed_val % 3;
            if (rem < 0) { score -= 1; rem += 3; }

            float dataset_score = static_cast<float>(score);
            float wdl_result = static_cast<float>(rem) / 2.0f;

            float p = 1.0f / (1.0f + std::exp(-eval)); // sigmoid
            float t_score = 1.0f / (1.0f + std::exp(-(dataset_score / k)));
            float target = (1.0f - lam) * t_score + lam * wdl_result;

            float diff = p - target;
            
            local_loss += diff * diff;
            float dLoss_dEval = (2.0f * diff * p * (1.0f - p)) / batch_size;

            // Write to thread-LOCAL gradients, no locks needed!
            for (int f = 0; f < features.count; ++f) {
                local_grads[features.indices[f]] += features.coeffs[f] * dLoss_dEval;
            }
        }

        #pragma omp critical
        {
            total_loss += local_loss;
            for (int f = 0; f < Extractor::NUM_FEATURES; ++f) {
                g_ptr[f] += local_grads[f];
            }
        }
    }
    
    return total_loss / batch_size;
}

// 3. Bind to Python
PYBIND11_MODULE(cpp_tuner, m) {
    m.def("process_batch_material", &process_batch_impl<MaterialExtractor>);
    m.def("process_batch_prf", &process_batch_impl<PRFExtractor>);
    m.def("process_batch_psqt", &process_batch_impl<PSTExtractor>);
    m.def("process_batch_kp", &process_batch_impl<KPExtractor>);
    m.def("process_batch_pp", &process_batch_impl<PPExtractor>);
    m.def("process_batch_ppxk", &process_batch_impl<PPxKExtractor>);
}
