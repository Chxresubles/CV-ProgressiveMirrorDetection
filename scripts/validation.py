import json
import pickle
import argparse
from pathlib import Path
from torchvision.transforms import v2
from cvprogressivemirrordetection.trainer import ModelTrainer
from cvprogressivemirrordetection.dataset import PMDDataset
from cvprogressivemirrordetection.constants import MEAN, STD, IMAGE_RESIZE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Progressive Mirror Detection model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/PMD",
        help="Path to input data folder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./output",
        help="Path to output folder",
    )
    args = parser.parse_args()

    out_path = Path(args.output_path)
    with open(out_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    validation_transform = v2.Compose(
        [
            v2.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )
    validation_dataset = PMDDataset(Path(args.data_path), transform=validation_transform)
    print(f"Validation dataset contains {len(validation_dataset)} images")

    validator = ModelTrainer(model)

    validation_metrics = validator.evaluate(validation_dataset)
    print(f"Model validation metrics = {validation_metrics}")

    with open(out_path / "validation_metrics.json", "w") as f:
        json.dump(validation_metrics, f, indent=2)
