import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import load_chess_dataset
from model import ChessOutcomeModel

# 1. Generate training data
print("Generating chess position data...")
X_np, y_np = load_chess_dataset()

# 2. Convert to PyTorch Tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# 3. Initialize architecture
model = ChessOutcomeModel()
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
print("Training your PyTorch neural network...")
epochs = 100

for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nSuccess training done.")

# --- Testing Model Predictions ---
print("\n--- Testing Model Predictions ---")

mock_winning_board = [0] * 64
mock_winning_board[0] = 9
mock_winning_board[1] = 5

mock_losing_board = [0] * 64
mock_losing_board[0] = -9
mock_losing_board[1] = -5

test_inputs = torch.tensor([mock_winning_board, mock_losing_board], dtype=torch.float32)

with torch.no_grad():
    predictions = model(test_inputs)

print(f"Prediction for White winning position: {predictions[0].item():.4f}")
print(f"Prediction for Black winning position: {predictions[1].item():.4f}")

if predictions[0].item() >= 0.5:
    print("[PASS] Effectiveness Check 1 Passed")
else:
    print("[FAIL] Check 1 Failed.")

if predictions[1].item() < 0.5:
    print("[PASS] Effectiveness Check 2 Passed")
else:
    print("[FAIL] Check 2 Failed.")