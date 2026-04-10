import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import struct

class ChessBinaryDataset(Dataset):
    def __init__(self, file_path):
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        assert len(raw_data) % 48 == 0, "File size is not a multiple of 48 bytes!"
        self.data = raw_data.reshape(-1, 48)
        self.num_positions = len(self.data)

    def __len__(self):
        return self.num_positions

    def __getitem__(self, idx):
        record = self.data[idx]
        
        # Parse Score and WDL
        packed_val = struct.unpack('<h', record[40:42].tobytes())[0]
        dataset_score = float(packed_val // 3)
        wdl_result = float(packed_val % 3) / 2.0
        
        # Parse Mailbox and extract nibbles
        mailbox = record[8:40]
        squares = np.empty(64, dtype=np.uint8)
        squares[0::2] = mailbox & 0x0F          
        squares[1::2] = (mailbox >> 4) & 0x0F     
        
        # Branchless piece counting
        counts = np.bincount(squares[squares != 15], minlength=12)
        stm_counts = counts[0:6]
        nstm_counts = counts[6:12]
        
        # Net Features [P, N, B, R, Q, K]
        net_features = stm_counts - nstm_counts
            
        return (
            torch.tensor(net_features, dtype=torch.float32),
            torch.tensor([dataset_score], dtype=torch.float32),
            torch.tensor([wdl_result], dtype=torch.float32)
        )

class MaterialTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(6, 1, bias=True)
        
        with torch.no_grad():
            self.weights.weight.copy_(
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            )
            self.weights.bias.fill_(0.0)

    def forward(self, features):
        return self.weights(features)
    
def blended_loss(eval_score, dataset_score, wdl_result, lam=0.5, k=400.0):
    pred_prob = torch.sigmoid(eval_score)
    dataset_prob = torch.sigmoid(dataset_score / k)
    target = (1.0 - lam) * dataset_prob + lam * wdl_result
    return torch.mean((pred_prob - target) ** 2)

if __name__ == "__main__":
    FILE_PATH = "data.bin"  # <--- Change this to your actual file
    
    # Load dataset
    dataset = ChessBinaryDataset(FILE_PATH)
    print(f"Loaded {len(dataset)} positions.")
    
    # Full-batch gradient descent: batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    model = MaterialTuner()
    all_wdls = []
    for _, _, wdl in dataloader:
        all_wdls.append(wdl)
    print("WDL Distribution:", torch.unique(torch.cat(all_wdls), return_counts=True))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000
    k = 400
    lam = 1.0
    
    print("\nStarting Training...")
    print(f"Initial Weights: {model.weights.weight.data.round().int().tolist()[0]}")
    
    # Since there is only 1 batch, we can just grab it once to save overhead
    features, dataset_score, wdl_result = next(iter(dataloader))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        current_eval = model(features)
        loss = blended_loss(current_eval, dataset_score, wdl_result, lam, k)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f}")
            
    print("\nTraining Complete!")
    print(f"Final Float Weights: {model.weights.weight.data.tolist()[0]}")
    print(f"Final Float Bias:  {model.weights.bias.data.item()}")
    
    raw_weights = model.weights.weight.data
    raw_bias = model.weights.bias.data
    quantized_weights = (raw_weights * k).round().int().tolist()[0]
    quantized_bias = (raw_bias * k).round().int().item()
    print(f"\nFinal Quantized Weights (Export to C++):")
    print(f"P: {quantized_weights[0]}")
    print(f"B: {quantized_weights[1]}")
    print(f"Q: {quantized_weights[2]}")
    print(f"N: {quantized_weights[3]}")
    print(f"R: {quantized_weights[4]}")
    print(f"K: {quantized_weights[5]}  <-- Should still be 0!")
    print(f"Tempo: {quantized_bias}")