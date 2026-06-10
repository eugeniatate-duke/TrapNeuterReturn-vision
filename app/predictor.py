from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision import models


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "cat",
    "dog",
    "opossum",
    "raccoon",
]


class Predictor:

    def __init__(self):

        self.model = (
            models.mobilenet_v3_small()
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            4,
        )

        self.model.load_state_dict(
            torch.load(
                "models/mobilenet_v3_small.pth",
                map_location=DEVICE,
            )
        )

        self.model.to(DEVICE)

        self.model.eval()

        self.transform = (
            transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
            ])
        )

    def predict(self, image):

        image = image.convert("RGB")

        tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            outputs = self.model(tensor)

            probs = torch.softmax(
                outputs,
                dim=1,
            )[0]

        confidence = (
            probs.max().item()
        )

        prediction = (
            probs.argmax().item()
        )

        return {
            "animal":
                CLASS_NAMES[prediction],
            "confidence":
                confidence,
            "probabilities":
                {
                    CLASS_NAMES[i]:
                    probs[i].item()
                    for i in range(4)
                },
        }