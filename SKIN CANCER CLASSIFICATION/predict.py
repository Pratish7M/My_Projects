import os
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
from dotenv import load_dotenv
import textwrap
import google.generativeai as genai
from werkzeug.utils import secure_filename
# Load environment variables from .env
load_dotenv()

# Configure Google Generative AI with API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model_path = './skin.keras'  # Adjust the path to your model
DermaNet = load_model(model_path)

# Define class labels
classes = {
    4: 'Nevus',
    6: 'Melanoma',
    2: 'Seborrheic Keratosis',
    1: 'Basal Cell Carcinoma',
    5: 'Vascular Lesion',
    0: 'Actinic Keratosis',
    3: 'Dermatofibroma'
}

# Image preprocessing function
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((28, 28))  # Adjust size as per your model
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def homepage():
    return render_template('index.html')

# Define the main route

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Secure the filename and create uploads directory if it doesn't exist
        filename = secure_filename(file.filename)
        uploads_dir = 'static/uploads'

        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        img_path = os.path.join(uploads_dir, filename)
        print(f"Saving image to: {img_path}")  # Debug print
        file.save(img_path)  # Save the file to the uploads directory

        # Open and preprocess the image
        img = Image.open(file)
        img_array = preprocess_image(img)

        # Make prediction
        prediction = DermaNet.predict(img_array)
        pred_label = np.argmax(prediction, axis=1)[0]
        pred_class = classes[pred_label]

        return render_template('result.html', class_name=pred_class, image_path=filename)

    return render_template('upload.html')




def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Home route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    response = ""
    if request.method == "POST":
        input_text = request.form["input"]
        response = get_gemini_response(input_text)
    return render_template("chat.html", response=response)




# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
