from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import BigramModel
from embedding import get_embedding

import torch
from fastapi import UploadFile, File
from PIL import Image
from torchvision import transforms
from cnn_model import CNN
import io

app = FastAPI()


cnn_model = CNN()
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
cnn_model.eval()


corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word, request.length
    )
    return {"generated_text": generated_text}

@app.get("/embedding")
def embedding(word: str):
    vector = get_embedding(word)
    return {"word": word, "embedding": vector}

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(image)
        prediction = torch.argmax(output, dim=1).item()

    return {"predicted_class": class_names[prediction]}