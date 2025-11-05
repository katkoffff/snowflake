from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import sam_model_registry, SamPredictor
import numpy as np, torch, cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="Snowflakes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_TYPE = "vit_b"
MODEL_PATH = "models/sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
sam.to(device)
predictor = SamPredictor(sam)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    image_np = np.array(img)
    predictor.set_image(image_np)

    h, w, _ = image_np.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = (masks[0] * 255).astype(np.uint8)
    _, mask_png = cv2.imencode(".png", mask)
    return {"mask": mask_png.tobytes()}
