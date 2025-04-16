import io
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
import pickle
import uvicorn
import torch
from PIL import Image
from torchvision.transforms import v2
from cvprogressivemirrordetection.constants import MEAN, STD, IMAGE_RESIZE

# Initialize FastAPI app
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading tensors on {DEVICE.type}")

TRANSFORMS = v2.Compose(
    [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

# Load the model
try:
    with open("./output/model.pkl", "rb") as f:
        model = pickle.load(f).to(device=DEVICE)
except FileNotFoundError:
    raise Exception("Model files not found")


@app.post("/predict")
async def predict(image: UploadFile):
    try:
        # Load input image
        img = Image.open(io.BytesIO(await image.read()))

        # Preprocess the data
        img = TRANSFORMS(img).to(device=DEVICE).unsqueeze(0)

        # Make prediction
        raw_segmentation = model(img)[0, 0].cpu().detach().numpy()
        mirror_segmentation = Image.fromarray(
            (raw_segmentation > 0.5).astype("uint8") * 255, "L"
        )

        bytes_io = io.BytesIO()
        mirror_segmentation.save(bytes_io, format="PNG")
        return Response(bytes_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
