import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Extracts activation maps from the final target convolutional layer
    to show where the model focuses during classification.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks to capture intermediate maps
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        # Default to the predicted class if none is passed
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Extract weights and feature maps
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Compute global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # Weight activation channels
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU on target heatmap, resize to standard input shape (224, 224)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        # Min-Max Normalization (0 to 1 scale)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam


def overlay_heatmap(img, heatmap, alpha=0.35):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    
    Parameters:
        img (np.ndarray): Original image as an RGB numpy array (224, 224, 3).
        heatmap (np.ndarray): Normalized Grad-CAM heatmap values (224, 224).
        alpha (float): Transparency multiplier for the heatmap blend (0.0 to 1.0).
                       Lower value (e.g. 0.3) = clearer original image.
                       Higher value (e.g. 0.6) = brighter heatmap colors.
    """
    # 1. Scale float heatmap (0.0 - 1.0) to uint8 (0 - 255)
    heatmap_bytes = np.uint8(255 * heatmap)
    
    # 2. Apply OpenCV JET colormap (returns BGR format)
    heatmap_colored = cv2.applyColorMap(heatmap_bytes, cv2.COLORMAP_JET)
    
    # 3. Convert BGR -> RGB format to align with Streamlit & PIL color space
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 4. Perform weighted linear blending between overlay and base image
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img