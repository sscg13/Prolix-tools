import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass

import cpp_tuner

# --- 1. Export Functions ---

def export_psqt(raw_weights):
    """
    Merges factorized PST weights (391 features)
    into a flat PST array (385 features) for engine evaluation.
    """
    merged_weights = np.zeros(385, dtype=np.int32)

    # Map Bias (Training index 0 -> Engine index 384)
    merged_weights[384] = raw_weights[0]

    # Merge Material and PST into Flat PST
    for p in range(6):
        mat_val = raw_weights[1 + p]
        for sq in range(64):
            pst_delta = raw_weights[7 + p * 64 + sq]
            merged_weights[p * 64 + sq] = mat_val + pst_delta

    return merged_weights

def export_kp(raw_weights):
    """
    Merges factorized KP weights into the flat bucketed form used by the engine.

    Raw layout (23233 = 704 + 32*704 + 1):
      [0 .. 703]       : P base weights, indexed as pt * 64 + sq
                         (11 piece types: 0-4 friendly, 5-9 enemy, 10 enemy king)
      [704 .. 23231]   : KP bucket weights, indexed as
                           704 + (k_idx * 11 + pt) * 64 + sq
                         for k_idx in [0, 32) (king bucket)
      [23232]          : STM-advantage bias

    Merged layout (22529 = 32*704 + 1):
      [0 .. 22527] : KP weights, indexed as (k_idx * 11 + pt) * 64 + sq,
                     with P base folded into every bucket.
      [22528]      : Bias.
    """
    merged = np.zeros(22529, dtype=np.int32)
    p_base  = raw_weights[:704]                            # (704,)
    kp_flat = raw_weights[704:23232].reshape(32, 704)      # (k_idx, pt*64+sq)
    merged[:22528] = (kp_flat + p_base[None, :]).reshape(-1)
    merged[22528]  = raw_weights[23232]
    return merged


# Lazy cache for the PP canonical pair-index map (built on first export).
_PP_PAIR_DATA = None
_PP_NUM_CANONICAL = 147072

def _build_pp_pair_data():
    """Mirrors the C++ PPInit logic so exported indices match training indices."""
    global _PP_PAIR_DATA
    if _PP_PAIR_DATA is not None:
        return _PP_PAIR_DATA
    N = 768
    pair_data = np.full(N * N, -1, dtype=np.int64)
    next_idx = 0
    for a in range(N):
        pt_a, sq_a = divmod(a, 64)
        fa = ((pt_a + 6) % 12) * 64 + (sq_a ^ 56)
        row_a = a * N
        for b in range(a + 1, N):
            if pair_data[row_a + b] != -1:
                continue
            pt_b, sq_b = divmod(b, 64)
            fb = ((pt_b + 6) % 12) * 64 + (sq_b ^ 56)
            a2, b2 = (fa, fb) if fa < fb else (fb, fa)
            if a == a2 and b == b2:
                pair_data[row_a + b] = -2
                pair_data[b * N + a] = -2
                continue
            code_pos = next_idx << 1
            code_neg = code_pos | 1
            pair_data[row_a + b]    = code_pos
            pair_data[b * N + a]    = code_pos
            pair_data[a2 * N + b2]  = code_neg
            pair_data[b2 * N + a2]  = code_neg
            next_idx += 1
    assert next_idx == _PP_NUM_CANONICAL, (
        f"PP canonical count mismatch: built {next_idx}, expected {_PP_NUM_CANONICAL}"
    )
    _PP_PAIR_DATA = pair_data
    return pair_data


def export_pp(raw_weights):
    """
    Expands the 147072 canonical PP weights (+ bias) into a dense 768x768
    matrix (+ bias) for the engine.

    Output layout (589825 = 768*768 + 1):
      [i * 768 + j] = weight of piece pair (i, j), where i,j in [0, 768).
                      Satisfies W[i][j] = W[j][i]
                        and      W[i][j] = -W[flip(i)][flip(j)],
                      with flip(f) = ((pt+6) % 12) * 64 + (sq ^ 56).
      [768 * 768]   = Bias.
    Self-symmetric pairs (those equal to their own flip as an unordered pair)
    are pinned to 0.
    """
    N = 768
    pair_data = _build_pp_pair_data()
    out = np.zeros(N * N + 1, dtype=np.int32)

    # Vectorised expansion.
    d = pair_data
    valid = d >= 0
    idx   = (d >> 1).astype(np.int64)
    sign  = np.where((d & 1) == 1, -1, 1).astype(np.int32)
    # Gather canonical weights for valid entries.
    flat  = np.zeros(N * N, dtype=np.int32)
    canon = raw_weights[:_PP_NUM_CANONICAL].astype(np.int32)
    flat[valid] = sign[valid] * canon[idx[valid]]
    out[:N * N]  = flat
    out[N * N]   = raw_weights[_PP_NUM_CANONICAL]
    return out




# --- 2. Dynamic Backend Registry ---
# Easily add new tuners here without changing the training loop
TUNER_BACKENDS = {
    "material": {
        "func": cpp_tuner.process_batch_material,
        "num_features": 7,  # 6 pieces + 1 bias
        "export_func": None,
    },
    "prf": {
        "func": cpp_tuner.process_batch_prf,
        "num_features": 97,  # 48 file + 48 rank + 1 bias
        "export_func": None,
    },
    "psqt": {
        "func": cpp_tuner.process_batch_psqt,
        "num_features": 391,  # 1 bias + 6 material + 6 pieces * 64 squares
        "export_func": export_psqt,
    },
    "kp": {
        "func": cpp_tuner.process_batch_kp,
        "num_features": 23233,  # 704 base + 32 * 704 buckets + 1 bias
        "export_func": export_kp,
    },
    "pp": {
        "func": cpp_tuner.process_batch_pp,
        "num_features": 147073,  # 147072 canonical pairs + 1 bias
        "export_func": export_pp,
    },
    "ppxk": {
        "func": cpp_tuner.process_batch_ppxk,
        "num_features": 593921,  # 768*768 PP + 64*64 K + 1 bias
        "export_func": None,
    },
}

# --- 3. Configuration Dataclass ---
@dataclass
class TunerConfig:
    file_path: str
    tuner_type: str
    epochs: int
    batch_size: int
    lr: float
    k: float
    lambda_val: float

# --- 4. The Tuner Class ---
class ChessEngineTuner:
    def __init__(self, config: TunerConfig):
        self.config = config

        if self.config.tuner_type not in TUNER_BACKENDS:
            raise ValueError(f"Unknown tuner type: {self.config.tuner_type}")

        self.backend = TUNER_BACKENDS[self.config.tuner_type]
        self.num_features = self.backend["num_features"]
        self.cpp_process_func = self.backend["func"]

        self._load_dataset()
        self._setup_optimizer()

    def _load_dataset(self):
        print(f"Loading dataset from {self.config.file_path} into RAM...")
        raw_data = np.fromfile(self.config.file_path, dtype=np.uint8)
        self.dataset_tensor = torch.from_numpy(raw_data)

        self.total_positions = len(raw_data) // 48
        print(f"Dataset contains {self.total_positions:,} positions.")

        # Create the index array for blazing fast zero-copy shuffling
        self.indices = torch.arange(self.total_positions, dtype=torch.int32)

    def _setup_optimizer(self):
        # Allocate Weights and explicitly allocate the gradient buffer for C++
        self.weights = torch.zeros(self.num_features, dtype=torch.float32, requires_grad=True)
        self.weights.grad = torch.zeros_like(self.weights)
        self.optimizer = optim.Adam([self.weights], lr=self.config.lr)

    def train(self):
        print(f"\nStarting Training [{self.config.tuner_type.upper()} Tuner]...")

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            positions_processed = 0
            start_time = time.perf_counter()

            # Shuffle indices at the start of every epoch
            self.indices = self.indices[torch.randperm(self.total_positions)]

            for start_idx in range(0, self.total_positions, self.config.batch_size):
                current_batch_size = min(self.config.batch_size, self.total_positions - start_idx)

                self.weights.grad.zero_()

                # The C++ Black Box
                batch_loss = self.cpp_process_func(
                    start_idx,
                    current_batch_size,
                    self.config.k,
                    self.config.lambda_val,
                    self.weights,
                    self.weights.grad,
                    self.dataset_tensor,
                    self.indices
                )

                self.optimizer.step()

                total_loss += batch_loss * current_batch_size
                positions_processed += current_batch_size

            # Logging
            elapsed_time = time.perf_counter() - start_time
            pos_per_sec = positions_processed / elapsed_time
            avg_loss = total_loss / self.total_positions

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:4d}/{self.config.epochs} | Loss: {avg_loss:.6f} | Speed: {pos_per_sec:,.0f} pos/sec")

        print("\nTraining Complete!")

    def export_weights(self):
        weights_np = self.weights.data.cpu().numpy()
        raw_weights = np.round(weights_np * self.config.k).astype(np.int32).flatten()

        t = self.config.tuner_type
        export_func = self.backend["export_func"]

        if export_func is None:
            if len(raw_weights) < 1000:
                print(f"\nRaw Weights ({t}):\n", raw_weights)
            else:
                raw_filename = f"{t}_weights_raw.bin"
                raw_weights.astype('<i4').tofile(raw_filename)
                print(f"\n[Success] {len(raw_weights)} raw weights exported to {raw_filename}")
                print(f"Format: Little-endian int32 (4 bytes per weight)")
        else:
            export_array = export_func(raw_weights)
            if len(raw_weights) < 1000:
                print(f"\nRaw Weights ({t}):\n", raw_weights)
                print(f"\nMerged Weights ({t}):\n", export_array)
            else:
                raw_filename = f"{t}_weights_raw.bin"
                merged_filename = f"{t}_weights_merged.bin"
                raw_weights.astype('<i4').tofile(raw_filename)
                export_array.astype('<i4').tofile(merged_filename)
                print(f"\n[Success] {len(raw_weights)} raw weights exported to {raw_filename}")
                print(f"[Success] {len(export_array)} merged weights exported to {merged_filename}")
                print(f"Format: Little-endian int32 (4 bytes per weight)")

# --- 5. CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance C++ Chess Tuner")

    parser.add_argument("--data", type=str, default="data.bin", help="Path to the binary dataset")
    parser.add_argument("--tuner", type=str, choices=TUNER_BACKENDS.keys(), default="material", help="Which feature extractor to use")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16384, help="Positions per batch")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--k", type=float, default=400.0, help="Sigmoid scaling factor")
    parser.add_argument("--lam", type=float, default=1.0, help="WDL interpolation lambda")

    args = parser.parse_args()

    config = TunerConfig(
        file_path=args.data,
        tuner_type=args.tuner,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k=args.k,
        lambda_val=args.lam
    )

    tuner = ChessEngineTuner(config)
    tuner.train()
    tuner.export_weights()
