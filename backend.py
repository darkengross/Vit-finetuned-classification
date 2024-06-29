from typing import List
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import io
from torchvision.transforms import InterpolationMode

# Transformation pipeline
data_transform = transforms.Compose([
    transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model(num_classes=6):
    model = models.vit_b_16(weights=None)
    model.heads[-1] = torch.nn.Linear(in_features=model.heads[-1].in_features, out_features=num_classes, bias=True)
    return model

def load_model(model_path, num_classes=6):
    model = get_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

def pre_process_dpt(image_bytes, transform):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_batch = transform(image).unsqueeze(0)
    return input_batch

@torch.no_grad()
def predict_image(image, model, classes=('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    image = image.to(device)
    logits = model(image)
    predicted_class = torch.argmax(logits, dim=1).item()
    return classes[predicted_class]

app = FastAPI()

model_path = './models/fine_tuned_vit_b_16_on_intel_image_dataset.pt'
model = load_model(model_path)

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    if model is None:
        return {"error": "Model failed to load. Please check the model path and try again."}
    
    results = []
    for file in files:
        contents = await file.read()
        try:
            input_batch = pre_process_dpt(contents, data_transform)
            class_type = predict_image(input_batch, model)
            results.append({"filename": file.filename, "class_type": class_type})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return results
