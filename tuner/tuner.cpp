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
    static constexpr int NUM_FEATURES = 199; // 1 Bias + 6 Material + 192 Mirrored PST
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
            
            // Horizontal Mirroring (Fold files E-H into A-D)
            int file = relative_sq % 8;
            int rank = relative_sq / 8;
            int mirrored_file = (file > 3) ? (7 - file) : file;
            int mirrored_sq = (rank * 4) + mirrored_file; // Range: 0 to 31
            
            // Calculate Indices
            int mat_idx = 1 + base_piece;
            int pst_idx = 7 + (base_piece * 32) + mirrored_sq;
            
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
}
