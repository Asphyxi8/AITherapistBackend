import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
import torch
import torch.nn.functional as F
from torchvision import transforms
import google.generativeai as palm
import torch.nn as nn
import logging
from werkzeug.security import generate_password_hash, check_password_hash
logging.basicConfig(level=logging.DEBUG)
# Number of emotion classes
num_classes = 7
import json

# Define the emotion detection model class
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)  # Adjust for the size of feature maps
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 6 * 6)  # Flatten the feature maps
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SECRET_KEY'] = 'supersecretkey'

# Configure JWT
app.config['JWT_SECRET_KEY'] = 'jwtsecretkey'  # Change this to a more secure secret key
jwt = JWTManager(app)

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load user by ID (for JWT)
def get_user_by_id(user_id):
    return User.query.get(user_id)

# Conversation model
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    messages = db.Column(db.Text, nullable=False, default = "[]")  # Store the whole conversation as JSON or text
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('conversations', lazy=True))

# Load the PyTorch model
emotion_model = EmotionModel()

# Load the state_dict
model_path = r"C:\Users\nandh\OneDrive\Pictures\Desktop\AITherapist\AITherapistBackend\ai-therapist-app\emotion_detection_model.pth"
emotion_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))


emotion_model.eval()  # Set the model to evaluation mode

# Define class labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Authenticate Google PaLM API
palm.configure(api_key='AIzaSyCuYPvvR8ZkxPHqB8TbUG2fE80VQBc95Yg')

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing transformations for the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Route to register a user
@app.route('/register', methods=['POST'])
def register():
    if not request.is_json:
        return jsonify({"message": "Invalid request. JSON data expected."}), 400

    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        if 'username' not in data or 'password' not in data:
            return jsonify({"message": "Missing required fields: username and password"}), 400

        username = data['username']
        password = data['password']

        if User.query.filter_by(username=username).first():
            return jsonify({"message": "User already exists!"}), 400

        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password)

        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User registered successfully!"}), 201
    except Exception as e:
        app.logger.error(f"Error while registering user: {str(e)}")
        db.session.rollback()
        return jsonify({"message": "Internal server error"}), 500


# Route to login and generate JWT token
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and user.password == data['password']:
        # Create JWT token
        access_token = create_access_token(identity=str(user.id))
        app.logger.info(f"Access token generated: {access_token}")
        return jsonify({"message": "Login successful!", "access_token": access_token})
    return jsonify({"message": "Invalid credentials!"}), 401

@app.route('/conversations', methods=['GET'])
def get_conversations():
    try:
        verify_jwt_in_request()  # Validates the JWT
        user_id = get_jwt_identity()  # Extracts user ID from the token
    except Exception as e:
        app.logger.error(f"Token verification failed: {str(e)}")
        return jsonify({"message": "Invalid token"}), 401

    user_conversations = Conversation.query.filter_by(user_id=user_id).all()
    app.logger.info(user_conversations)
    return jsonify([{"id": conv.id, "title": conv.title, "messages": conv.messages} for conv in user_conversations])

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
    except Exception as e:
        app.logger.error(f"Token verification failed: {str(e)}")
        return jsonify({"message": "Invalid token"}), 401

    data = request.json
    new_conv = Conversation(user_id=user_id, title=data['title'], messages="")
    db.session.add(new_conv)
    db.session.commit()
    return jsonify({"message": "New conversation started!", "conversation_id": new_conv.id})
@app.route("/")
def home():
    return "Welcome to AI Therapist App!"
@app.route('/conversation/<int:conversation_id>', methods=['GET'])
@jwt_required()
def get_conversation(conversation_id):
    user_id = get_jwt_identity()
    conversation = Conversation.query.get_or_404(conversation_id)

    # if conversation.user_id != int(user_id):
    #     return jsonify({"message": "Unauthorized"}), 403
    try:
        conversation_data = json.loads(conversation.messages)
    except (json.JSONDecodeError, TypeError):
        conversation_data = []
    return jsonify({
        "id": conversation.id,
        "title": conversation.title,
        "messages": conversation_data
    })


@app.route('/conversation/<int:conversation_id>', methods=['POST'])
def continue_conversation(conversation_id):
    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
    except Exception as e:
        app.logger.error(f"Token verification failed: {str(e)}")
        return jsonify({"message": "Invalid token"}), 401

    data = request.json
    message = data['message']
    snapshot_data = data['snapshot']

    # Find the conversation
    conversation = Conversation.query.get_or_404(conversation_id)
    # if conversation.user_id != int(user_id):
    #     return jsonify({"message": "Unauthorized"}), 403

    # Decode the snapshot image (Base64)
    snapshot_bytes = base64.b64decode(snapshot_data.split(',')[1])
    np_img = np.frombuffer(snapshot_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect faces in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are found, return an error message
    if len(faces) == 0:
        return jsonify({"response": "No face detected in the image."})

    # Assume we're working with the first detected face
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]

    # Preprocess the face image
    input_image = preprocess(face).unsqueeze(0)  # Add batch dimension

    # Predict emotion
    with torch.no_grad():
        output = emotion_model(input_image)
        probabilities = F.softmax(output, dim=1)
        emotion_index = torch.argmax(probabilities, dim=1).item()
        emotion = emotion_labels[emotion_index]

    # Initialization prompt for therapist AI
    initialization_prompt = (
        "You are a therapist AI trained in multiple therapeutic approaches, including Cognitive Behavioral Therapy (CBT), "
        "Dialectical Behavioral Therapy (DBT), Mindfulness-Based Stress Reduction (MBSR), Acceptance and Commitment Therapy (ACT), "
        "Person-Centered Therapy, Psychodynamic Therapy, and more. Based on the user's emotional state, apply the appropriate "
        "technique. For instance, use CBT to challenge negative thoughts, DBT for emotional regulation, mindfulness for stress "
        "reduction, and ACT for acceptance and behavior change. Ensure responses are empathetic, trauma-informed, and tailored "
        "to long-term well-being. You are speaking to a user who is feeling {emotion}."
    ).format(emotion=emotion)

    # Generate the full prompt
    full_prompt = f"Here is your message history with the user: Initiliaization: {initialization_prompt} {conversation.messages}\n\nUser now said: {message}. Give 1 appropriate response."
    model = palm.GenerativeModel(model_name='gemini-1.5-flash')

    # Generate a response from PaLM API using the generate_text method
    response = model.generate_content(full_prompt)

    try:
        conversation_data = json.loads(conversation.messages)
    except (json.JSONDecodeError, TypeError):
        # If messages is empty or invalid JSON, initialize as empty list
        conversation_data = []
        app.logger.info("Initializing empty conversation data")
        # Append new messages
    conversation_data.append({"role": "user", "message": message, "emotion": emotion})
    conversation_data.append({"role": "ai", "message": response.text})            # Convert back to JSON string and save
    conversation.messages = json.dumps(conversation_data)
    db.session.commit()

    # Send back the updated conversation
    return jsonify({"response": response.text, "updated_messages": conversation_data})

if __name__ == '__main__':
    with app.app_context():  # Ensure we're inside the app context
        db.create_all()
    app.run(debug=True)
