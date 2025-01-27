# inference_seq.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import time

# Reuse the model definition
class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # up to avgpool
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x, h_in=None):
        """
        x: [batch, seq_len, 3, 224, 224]
        h_in: optional (h, c) for LSTM
        returns: (twist, (h, c))
        """
        batch_size, seq_len, C, H, W = x.shape
        # Flatten
        x = x.view(batch_size * seq_len, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(batch_size, seq_len, 512)

        if h_in is not None:
            outputs, (h_out, c_out) = self.lstm(feats, h_in)
        else:
            outputs, (h_out, c_out) = self.lstm(feats)

        # Take last output
        last_output = outputs[:, -1, :]  # [batch, hidden_size]
        twist = self.fc(last_output)     # [batch, 3]
        return twist, (h_out, c_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth of trained model')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    args = parser.parse_args()

    # Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Inference on device: {device}")

    # Build model
    model = CNNLSTMModel(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # We'll do a simple loop that simulates reading frames from disk or a camera
    # For demonstration, we just keep re-inferencing the same image
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"Warning: {test_image_path} not found. Using a random image if you have one.")
    print("Starting inference loop...")

    # Initialize LSTM hidden state to None
    h_state = None

    for step in range(10):
        # In real scenario, grab a fresh frame from the camera
        image = Image.open(test_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(device)
        # shape => [batch=1, seq_len=1, 3, 224, 224]

        with torch.no_grad():
            twist_pred, h_state = model(image_tensor, h_state)
            # twist_pred shape: [1, 3]
            # h_state is (h, c) each with shape [num_layers, 1, hidden_size]

        twist_pred = twist_pred.squeeze().cpu().numpy()  # (3,)
        print(f"Step {step} -> Predicted twist: x={twist_pred[0]:.3f}, y={twist_pred[1]:.3f}, theta={twist_pred[2]:.3f}")

        # Sleep to simulate real-time
        time.sleep(0.1)

    print("Inference complete.")

if __name__ == "__main__":
    main()
