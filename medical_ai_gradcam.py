# medical_ai_gradcam.py

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# 2. LOAD RESNET50
# ==========================================

print("\nLoading ResNet50 model...")

model = models.resnet50(weights=None)

model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint_path = "medical_ai_resnet50.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove DataParallel prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key.replace("module.", "")

    clean_state_dict[new_key] = value


model.load_state_dict(clean_state_dict)

model.to(device)

model.eval()

print("ResNet50 loaded successfully!")


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. GRAD-CAM CLASS
# ==========================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.save_activations
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradients
        )


    def save_activations(self, module, input, output):

        self.activations = output


    def save_gradients(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]


    def generate(self, image_tensor, target_class):

        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[0, target_class]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        # Average gradients across spatial dimensions
        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        # Weighted combination of activation maps
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, weight in enumerate(weights):

            cam += weight * activations[i]

        # ReLU: keep positive contributions
        cam = torch.relu(cam)

        # Normalize heatmap
        cam = cam - cam.min()

        if cam.max() != 0:

            cam = cam / cam.max()

        cam = cam.detach().cpu().numpy()

        return cam


# ==========================================
# 5. SELECT TARGET LAYER
# ==========================================

target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)


# ==========================================
# 6. ENTER IMAGE PATH
# ==========================================

print("\n" + "=" * 50)

print("GRAD-CAM IMAGE ANALYSIS")

print("=" * 50)

image_path = input(
    "\nEnter the path of a chest X-ray image: "
).strip().strip('"')


if not os.path.exists(image_path):

    print("\nERROR: Image not found!")

    exit()


# ==========================================
# 7. LOAD IMAGE
# ==========================================

original_image = Image.open(image_path).convert("RGB")

input_tensor = transform(original_image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# ==========================================
# 8. MODEL PREDICTION
# ==========================================

with torch.no_grad():

    prediction = model(input_tensor)

    probabilities = torch.softmax(
        prediction,
        dim=1
    )


predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()


class_names = [
    "NORMAL",
    "PNEUMONIA"
]


predicted_label = class_names[predicted_class]

normal_probability = probabilities[0][0].item()

pneumonia_probability = probabilities[0][1].item()


print("\n" + "=" * 50)

print("PREDICTION RESULTS")

print("=" * 50)

print("Predicted class:", predicted_label)

print(f"NORMAL probability: {normal_probability * 100:.2f}%")

print(f"PNEUMONIA probability: {pneumonia_probability * 100:.2f}%")


# ==========================================
# 9. GENERATE GRAD-CAM
# ==========================================

print("\nGenerating Grad-CAM heatmap...")

heatmap = gradcam.generate(
    input_tensor,
    predicted_class
)


# ==========================================
# 10. RESIZE HEATMAP
# ==========================================

heatmap_image = Image.fromarray(
    np.uint8(heatmap * 255)
)

heatmap_image = heatmap_image.resize(
    original_image.size,
    Image.Resampling.BILINEAR
)

heatmap = np.array(heatmap_image) / 255.0


# ==========================================
# 11. CREATE OVERLAY
# ==========================================

original_array = np.array(original_image)

colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]

overlay = (
    0.6 * original_array / 255.0
    +
    0.4 * colored_heatmap
)

overlay = np.clip(
    overlay,
    0,
    1
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(original_image)

plt.title("Original Chest X-Ray")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(heatmap, cmap="jet")

plt.title("Grad-CAM Heatmap")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(overlay)

plt.title(
    f"Grad-CAM Overlay\nPrediction: {predicted_label}"
)

plt.axis("off")


plt.tight_layout()


# ==========================================
# 13. SAVE RESULT
# ==========================================

output_path = "medical_ai_resnet50_gradcam.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nGrad-CAM image saved as:")

print(output_path)


plt.show()


print("\nGrad-CAM analysis completed!")