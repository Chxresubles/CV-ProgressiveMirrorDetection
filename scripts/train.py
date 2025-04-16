import os
import json
import pickle
import argparse
from pathlib import Path
from torchvision.transforms import v2
from cvprogressivemirrordetection.trainer import ModelTrainer
from cvprogressivemirrordetection.dataset import PMDDataset
from cvprogressivemirrordetection.models import FCN, DeepLabV3, LRASPP, SAM2UNet
from cvprogressivemirrordetection.constants import MEAN, STD, IMAGE_RESIZE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Progressive Mirror Detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FCN",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model",
    )
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
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./pretrained_models",
        help="Path to pretrained model weights folder",
    )
    args = parser.parse_args()

    seed = 42

    if args.model == "FCN":
        model = FCN()
    elif args.model == "DeepLabV3":
        model = DeepLabV3()
    elif args.model == "LRASPP":
        model = LRASPP()
    elif args.model == "SAM2-UNet":
        if os.path.isfile(os.path.join(args.weights_path, "SAM2UNet-PMD.pth")):
            print("Using pretrained SAM2-UNet from weights file")
            model = SAM2UNet(os.path.join(args.weights_path, "SAM2UNet-PMD.pth"))
        else:
            print("Training SAM2-UNet from scratch")
            model = SAM2UNet()
    else:
        raise NotImplementedError(
            f"The selected model '{args.model}' is not supported. Select a model from the list of available model."
        )

    trainer = ModelTrainer(model)

    train_transform = v2.Compose(
        [
            v2.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )
    test_transform = v2.Compose(
        [
            v2.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )

    train_dataset = PMDDataset(
        Path(args.data_path) / "train", transform=train_transform
    )
    print(f"Train dataset contains {len(train_dataset)} images")
    test_dataset = PMDDataset(Path(args.data_path) / "test", transform=test_transform)
    print(f"Test dataset contains {len(test_dataset)} images")

    train_metrics = trainer.train(train_dataset, n_epochs=args.epochs, seed=seed)
    print(f"Model train metrics = {train_metrics}")

    test_metrics = trainer.evaluate(test_dataset)
    print(f"Model test metrics = {test_metrics}")

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(out_path / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(out_path / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
