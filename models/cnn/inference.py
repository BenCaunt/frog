import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

class ImageToTwistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.backbone(x)

def main():
    # Check for device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Load model
    model = ImageToTwistModel().to(device)
    model.load_state_dict(torch.load("robot_twist_model.pth", map_location=device))
    model.eval()

    # Example transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Suppose we have a constant inference time (seconds) to simulate
    inference_time = 0.05  # e.g. 50ms
    # We'll keep a buffer of predicted twists that covers 2x the inference time
    # In real robotics code, you'd pass these commands to a controller

    # Dummy loop: pretend we get 10 frames from somewhere
    predicted_commands_buffer = []
    for i in range(10):
        # In a real system you'd grab a frame from a camera
        # For demo, just load a test image or synthetic image
        image_path = "test_image.jpg"  # Replace with actual
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            twist_pred = model(input_tensor)  # shape (1, 3)
            twist_pred = twist_pred.squeeze().cpu().numpy()  # (3,)

        # Store the predicted (x, y, theta)
        predicted_commands_buffer.append(twist_pred)

        # Print or log
        print(f"Frame {i}, predicted twist: x={twist_pred[0]:.3f}, "
              f"y={twist_pred[1]:.3f}, theta={twist_pred[2]:.3f}")

        # Sleep to simulate real-time loop at ~20Hz
        time.sleep(inference_time)

    # Now you have a buffer of predicted commands you could send
    # to your real robot. In real code, you’d handle the logic for
    # “consuming” these commands at the correct rate or only the
    # newest command, etc.
    print("Inference complete. Predicted commands buffer:")
    for cmd in predicted_commands_buffer:
        print(cmd)

if __name__ == "__main__":
    main()
