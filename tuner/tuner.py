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

_PP_HALF = 384
_PP_SS   = _PP_HALF * (_PP_HALF - 1) // 2  # 73536

def _pp_flip(f):
    pt, sq = f // 64, f % 64
    return ((pt + 6) % 12) * 64 + (sq ^ 56)

def export_pp(raw_weights):
    """
    Decompresses canonical PP weights (147,073 = 2*C(384,2)+1) into a full
    768*768+1 array for direct engine lookup.

    Each entry out[fi*768 + fj] = sign * w[canonical_index(fi, fj)].
    Self-symmetric pairs (a == bp in the mixed case) stay 0.
    The last element is the bias.
    """
    out = np.zeros(768 * 768 + 1, dtype=np.int32)
    out[-1] = raw_weights[-1]  # bias

    for fi in range(768):
        for fj in range(768):
            if fi == fj:
                continue

            si = fi < _PP_HALF
            sj = fj < _PP_HALF
            fi_c, fj_c = fi, fj
            sign = 1

            if si == sj:
                if not si:                   # both NSTM: fold via vertical flip
                    fi_c = _pp_flip(fi_c)
                    fj_c = _pp_flip(fj_c)
                    sign = -1
                a, b = min(fi_c, fj_c), max(fi_c, fj_c)
                idx = b * (b - 1) // 2 + a
            else:                            # mixed STM/NSTM
                if not si:
                    fi_c, fj_c = fj, fi      # fi_c <- STM, fj_c <- NSTM
                a  = fi_c
                bp = _pp_flip(fj_c)
                if a == bp:
                    continue                  # self-symmetric, stays 0
                if a > bp:
                    a, bp = bp, a
                    sign = -1
                idx = _PP_SS + bp * (bp - 1) // 2 + a

            out[fi * 768 + fj] = sign * raw_weights[idx]

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
        "num_features": 147073,  # 2 * C(384,2) canonical pairs + 1 bias
        "export_func": export_pp,
        "init_func": cpp_tuner.init_pp_table,
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
        # Run any one-time backend initialisation (e.g. PP lookup table)
        init_fn = self.backend.get("init_func")
        if init_fn is not None:
            init_fn()

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
                raw_weights.astype(np.int16).astype('<i2').tofile(raw_filename)
                print(f"\n[Success] {len(raw_weights)} raw weights exported to {raw_filename}")
                print(f"Format: Little-endian int16 (2 bytes per weight)")
        else:
            export_array = export_func(raw_weights)
            if len(raw_weights) < 1000:
                print(f"\nRaw Weights ({t}):\n", raw_weights)
                print(f"\nMerged Weights ({t}):\n", export_array)
            else:
                raw_filename = f"{t}_weights_raw.bin"
                merged_filename = f"{t}_weights_merged.bin"
                raw_weights.astype(np.int16).astype('<i2').tofile(raw_filename)
                export_array.astype(np.int16).astype('<i2').tofile(merged_filename)
                print(f"\n[Success] {len(raw_weights)} raw weights exported to {raw_filename}")
                print(f"[Success] {len(export_array)} merged weights exported to {merged_filename}")
                print(f"Format: Little-endian int16 (2 bytes per weight)")

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
