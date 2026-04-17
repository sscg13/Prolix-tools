import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass

import cpp_tuner 

# --- 1. Dynamic Backend Registry ---
# Easily add new tuners here without changing the training loop
TUNER_BACKENDS = {
    "material": {
        "func": cpp_tuner.process_batch_material,
        "num_features": 7  # 6 pieces + 1 bias
    },
    "prf": {
        "func": cpp_tuner.process_batch_prf,
        "num_features": 97 # 48 file + 48 rank + 1 bias
    },
    "psqt": {
        "func": cpp_tuner.process_batch_psqt,
        "num_features": 385 # 6 pieces * 64 squares + 1 bias
    },
    "kp": {
        "func": cpp_tuner.process_batch_kp,
        "num_features": 23233 # 704 base + 32 * 704 buckets + 1 bias
    }
}

# --- 2. Configuration Dataclass ---
@dataclass
class TunerConfig:
    file_path: str
    tuner_type: str
    epochs: int
    batch_size: int
    lr: float
    k: float
    lambda_val: float

# --- 3. The Tuner Class ---
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
        quantized_weights = np.round(weights_np * self.config.k).astype(np.int32).flatten()
        num_params = len(quantized_weights)

        if num_params < 1000:
            print(f"\nFinal Quantized Weights ({self.config.tuner_type}):\n", quantized_weights)
        else:
            filename = f"{self.config.tuner_type}_weights.bin"
            quantized_weights.astype('<i4').tofile(filename)
            print(f"\n[Success] {num_params} weights exported to {filename}")
            print(f"Format: Little-endian int32 (4 bytes per weight)")

class EvalExporter:
    def __init__(self, scale):
        self.scale = scale

    def export_material(self, raw_weights):
        return raw_weights

    def export_prf(self, raw_weights):
        return raw_weights
    
    def export_pst(self, raw_weights):
        """
        Unmirrors and merges factorized PST weights (199 features) 
        into a flat PST array (385 features) for engine evaluation.
        """
        merged_weights = np.zeros(385, dtype=np.int32)
        
        # 1. Map Bias (Training index 0 -> Engine index 384)
        merged_weights[384] = raw_weights[0]
        
        # 2. Merge Material and Mirrored PST into Flat PST
        for p in range(6):
            mat_val = raw_weights[1 + p]
            for sq in range(64):
                file = sq % 8
                rank = sq // 8
                
                # Map full file (0-7) to mirrored file (0-3)
                mirrored_file = 7 - file if file > 3 else file
                mirrored_sq = rank * 4 + mirrored_file
                
                pst_delta = raw_weights[7 + p * 32 + mirrored_sq]
                
                # Engine array format: (piece * 64) + sq
                merged_weights[p * 64 + sq] = mat_val + pst_delta
                
        return merged_weights

    def export_weights(self, weights):
        # 1. Extract and Quantize
        weights_np = weights.data.cpu().numpy()
        raw_weights = np.round(weights_np * self.scale).astype(np.int32).flatten()
        
        # 2. Dispatch Logic (Transform to engine-ready format)
        # You can also trigger this via self.config.tuner_type == 'PST' if you prefer
        if len(raw_weights) == 199: 
            export_array = self.export_pst(raw_weights)
        # elif len(raw_weights) == ... : 
        #     export_array = self.export_kpst(raw_weights)
        else:
            export_array = raw_weights

        # 3. Output / File I/O
        num_params = len(export_array)

        if num_params < 1000:
            print(f"\nFinal Quantized Weights ({self.config.tuner_type}):\n", export_array)
        else:
            filename = f"{self.config.tuner_type}_weights.bin"
            export_array.astype('<i4').tofile(filename)
            print(f"\n[Success] {num_params} weights exported to {filename}")
            print(f"Format: Little-endian int32 (4 bytes per weight)")

        return export_array

# --- 4. CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance C++ Chess Tuner")
    
    parser.add_argument("--data", type=str, default="data.bin", help="Path to the binary dataset")
    parser.add_argument("--tuner", type=str, choices=TUNER_BACKENDS.keys(), default="material", help="Which feature extractor to use")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16384, help="Positions per batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
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