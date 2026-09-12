"""EfficientNetB0 oil-like versus look-alike inference utilities.

This module intentionally does not treat ImageNet weights as an oil-spill
classifier. Inference is enabled only when the caller supplies a checkpoint
fine-tuned for the two classes below.
"""

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local environment
    torch = None
    nn = None
    models = None
    transforms = None
    Image = None
    _IMPORT_ERROR = exc


CLASS_NAMES = ("Look-alike", "Oil-like")


class ClassifierLoadError(RuntimeError):
    """Raised when the optional classifier cannot be used safely."""


def runtime_status():
    """Return dependency/device information for transparent UI reporting."""

    if torch is None or models is None or transforms is None or Image is None:
        return {
            "available": False,
            "device": "Unavailable",
            "reason": f"PyTorch/torchvision unavailable: {_IMPORT_ERROR}",
        }

    if torch.cuda.is_available():
        device_name = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "MPS"
    else:
        device_name = "CPU"

    return {"available": True, "device": device_name, "reason": None}


def select_device():
    """Select CUDA first, then Apple MPS, then CPU."""

    if torch is None:
        raise ClassifierLoadError("PyTorch is not installed.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(use_imagenet_weights=False):
    """Build EfficientNetB0 with a two-class output layer.

    ImageNet weights are used only by the training script. The Streamlit
    inference path calls this with ``False`` and then loads an oil-spill
    checkpoint, so ImageNet predictions are never presented as oil results.
    """

    if torch is None or models is None or nn is None:
        raise ClassifierLoadError(
            "PyTorch and torchvision are required for EfficientNetB0."
        )

    imagenet_weights = (
        models.EfficientNet_B0_Weights.DEFAULT if use_imagenet_weights else None
    )
    model = models.efficientnet_b0(weights=imagenet_weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(CLASS_NAMES))
    return model


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ClassifierLoadError("The checkpoint is not a state-dict dictionary.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ClassifierLoadError("The checkpoint does not contain model weights.")

    # Support checkpoints saved from DataParallel without changing the model.
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


class EfficientNetOilClassifier:
    """Loaded EfficientNetB0 inference wrapper for candidate crops."""

    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device
        self.transform = models.EfficientNet_B0_Weights.DEFAULT.transforms()

    @classmethod
    def from_weights(cls, weights_path):
        """Load a locally supplied, fine-tuned two-class checkpoint."""

        if torch is None or models is None or transforms is None or Image is None:
            raise ClassifierLoadError(
                "Install torch, torchvision, and Pillow before loading the classifier."
            )

        path = Path(weights_path)
        if not path.is_file():
            raise ClassifierLoadError(f"Classifier weights not found: {path}")

        device = select_device()
        model = build_model(use_imagenet_weights=False)

        try:
            try:
                checkpoint = torch.load(
                    path,
                    map_location=device,
                    weights_only=True,
                )
            except TypeError:
                # Compatibility with older PyTorch versions without weights_only.
                checkpoint = torch.load(path, map_location=device)

            checkpoint_classes = (
                checkpoint.get("class_names")
                if isinstance(checkpoint, dict)
                else None
            )
            if checkpoint_classes is not None and tuple(checkpoint_classes) != CLASS_NAMES:
                raise ClassifierLoadError(
                    "Classifier class order does not match the required mapping: "
                    f"{CLASS_NAMES}. Found {checkpoint_classes}."
                )

            state_dict = _extract_state_dict(checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict,
                strict=False,
            )
            if missing_keys or unexpected_keys:
                raise ClassifierLoadError(
                    "Classifier weights are incompatible with EfficientNetB0: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
        except ClassifierLoadError:
            raise
        except Exception as exc:
            raise ClassifierLoadError(f"Could not load classifier weights: {exc}") from exc

        return cls(model, device)

    def predict(self, grayscale_crop):
        """Return class probabilities for one grayscale candidate crop."""

        if grayscale_crop is None or getattr(grayscale_crop, "size", 0) == 0:
            raise ValueError("Candidate crop is empty.")
        if grayscale_crop.ndim != 2:
            raise ValueError("Candidate crop must be a 2D grayscale array.")

        crop_uint8 = np.asarray(grayscale_crop, dtype=np.uint8)
        rgb_crop = np.repeat(crop_uint8[:, :, np.newaxis], 3, axis=2)
        image = Image.fromarray(rgb_crop, mode="RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]

        probabilities = probabilities.detach().cpu().numpy()
        predicted_index = int(np.argmax(probabilities))
        return {
            "look_alike_probability": float(probabilities[0]),
            "oil_probability": float(probabilities[1]),
            "classification": CLASS_NAMES[predicted_index],
        }
