import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for CNN-based models.
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):

        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, image, class_idx=None):

        self.model.eval()

        output = self.model(image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        output[:, class_idx].backward()

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1)

        # ReLU
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze()

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()