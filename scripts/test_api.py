import io
import argparse
import requests
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cvprogressivemirrordetection.constants import IMAGE_RESIZE


def display_image_and_pred_mask(image, pred_mask):
    image = np.array(image)
    pred_mask = np.array(pred_mask)

    plt.figure(layout="compressed", figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask)

    plt.subplot(1, 3, 3)
    masked_image = cv2.addWeighted(
        image, 0.5, pred_mask * np.array([1, 0, 0], dtype=np.uint8), 0.5, 0.0
    )
    plt.imshow(masked_image)

    plt.suptitle(f"Input image and prediction mask")
    plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Test Progressive Mirror Detection API"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="API endpoint URI (default: http://127.0.0.1:8000/predict)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="./data/PMD/test/ADE20K/image/ADE_train_00000082.jpg",
        help="Input image to send (default: ./data/PMD/test/ADE20K/image/ADE_train_00000082.jpg)",
    )
    args = parser.parse_args()

    try:
        # Send POST request to API
        with open(args.image_path, "rb") as f:
            response = requests.post(
                args.uri, files={"image": ("test_image.png", f, "image/png")}
            )

        # Check response
        if response.status_code == 200:
            print("Success!")
            print("Displaying response...")
            img = Image.open(args.image_path)
            pred = Image.open(io.BytesIO(response.content)).convert("RGB")
            display_image_and_pred_mask(img.resize((IMAGE_RESIZE, IMAGE_RESIZE)), pred)
        else:
            print(f"Error: Status code {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")


if __name__ == "__main__":
    main()
