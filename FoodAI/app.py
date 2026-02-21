from flask import Flask, render_template, request
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load your pretrained model (ResNet50 as before) ---
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dummy nutrition data function (replace with real API/model later)
def analyze_food(image_path, goal):
    # For now, just return dummy values
    food = "Rice"
    calories = 200
    protein = 4
    carbs = 45
    fat = 1

    suggestion = ""
    score = 0

    if goal == "weight_loss":
        if carbs > 30:
            suggestion = "High in carbs for weight loss. Add protein."
            score = 5
        else:
            suggestion = "Good choice for weight loss!"
            score = 8
    elif goal == "muscle_gain":
        if protein < 10:
            suggestion = "Low protein. Add eggs or chicken."
            score = 4
        else:
            suggestion = "Great for muscle gain!"
            score = 9
    elif goal == "diabetic":
        if carbs > 25:
            suggestion = "High carbs. Not ideal for diabetic diet."
            score = 5
        else:
            suggestion = "Suitable for diabetic diet."
            score = 8
    elif goal == "balanced":
        suggestion = "Moderately balanced meal."
        score = 7

    # --- Run AI model prediction (optional, just dummy prediction for now) ---
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    prediction = predicted.item()

    return food, calories, protein, carbs, fat, score, suggestion, prediction

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Form submission route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Get user goal
        goal = request.form.get('goal', 'balanced')

        # Analyze food
        food, calories, protein, carbs, fat, score, suggestion, prediction = analyze_food(filepath, goal)

        return render_template('result.html',
                       image_path=filepath,  # This is needed for the <img> tag
                       food=food,
                       calories=calories,
                       protein=protein,
                       carbs=carbs,
                       fat=fat,
                       score=score,
                       suggestion=suggestion,
                       prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)