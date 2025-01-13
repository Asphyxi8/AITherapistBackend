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
logging.basicConfig(level=logging.DEBUG)
# Number of emotion classes
num_classes = 7
import json

def parse_model_response(response_text):
    """
    Parse the model's response, which is expected to return JSON wrapped in ```json ... ```
    
    Args:
        response_text (str): The raw response string from the model.
    
    Returns:
        dict: Parsed JSON object if valid, or a default error response.
    """
    try:
        # Find the starting and ending indices of the JSON block
        json_start = response_text.find("```json")
        json_end = response_text.rfind("```")
        
        if json_start != -1 and json_end != -1:
            # Extract the JSON portion, excluding the ```json tags
            json_content = response_text[json_start + len("```json"):json_end].strip()
            # Parse the JSON content
            return json.loads(json_content)
        else:
            raise ValueError("JSON block not found in response")
    
    except (json.JSONDecodeError, ValueError) as e:
        # Log error for debugging
        app.logger.error(f"Failed to parse JSON response: {e}")
        # Return a default fallback response
        return {
            "answer": "I'm sorry, I couldn't process the response. Can we try again?",
            "therapy": {
                "name": "N/A",
                "reason": "Unable to determine due to parsing error.",
                "description": "N/A"
            }
        }
 
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
app.config['JWT_SECRET_KEY'] = 'jwtsecretkey'  # Change this to a more secure secret key
jwt = JWTManager(app)
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Indicates if the user is an admin

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


class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.String(300), nullable=True)
    format = db.Column(db.JSON, nullable=False)  # Stores test structure (questions, options, scoring rules)

    def get_scoring_rule(self):
        return self.format.get("scoring_rule", "sum")  # Default is summation


# Define UserTest model (to track test scores for users)
class UserTest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    score = db.Column(db.Integer, default=-1)
    user = db.relationship('User', backref=db.backref('user_tests', lazy=True))
    test = db.relationship('Test', backref=db.backref('user_tests', lazy=True))


@app.route('/api/test/<int:test_id>', methods=['GET'])
@jwt_required()
def get_test(test_id):
    # Fetch test by ID
    app.logger.info("HELLO")
    test = Test.query.get_or_404(test_id)
    app.logger.info(test)
    # You can return the test format and other details
    return jsonify({
        "id": test.id,
        "name": test.name,
        "description": test.description,
        "format": test.format
    })

@app.route('/api/submit/<int:test_id>', methods=['POST'])
@jwt_required()
def submit_test_results(test_id):
    user_id = get_jwt_identity()

    # Get the data sent with the request (test answers)
    answers = request.json.get('answers')  # Should be a list of answers

    # Calculate the score based on the test scoring rule
    test = Test.query.get_or_404(test_id)
    scoring_rule = test.get_scoring_rule()

    # Calculate score based on answers (for now, let's assume summing the scores)
    score = 0
    if scoring_rule == 'sum':
        for answer in answers:
            score += answer['score']
    elif scoring_rule == 'average':
        score = sum(answer['score'] for answer in answers) / len(answers)

    # Check if this user has already completed this test
    user_test = UserTest.query.filter_by(user_id=user_id, test_id=test_id).first()
    if user_test:
        # Update the score if the user has already taken the test
        user_test.score = score
    else:
        # Create a new UserTest record
        user_test = UserTest(user_id=user_id, test_id=test_id, score=score)
        db.session.add(user_test)

    db.session.commit()

    return jsonify({"message": "Test submitted successfully", "score": score}), 200


@app.route('/get_tests', methods=['GET'])
def get_tests():
    """
    Fetch all tests from the database with their details.
    """
    tests = Test.query.all()
    test_data = [
        {
            'id': test.id,
            'name': test.name,
            'description': test.description,
            'format': test.format
        }
        for test in tests
    ]
    return jsonify(test_data), 200

# Load the PyTorch model
emotion_model = EmotionModel()

# Load the state_dict
model_path = r"C:\Users\priya\Downloads\emotion_detection_model.pth"
emotion_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    data = request.json
    username = data['username']
    password = data['password']
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "User already exists!"}), 400
    user = User(username=username, password=password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!"})

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
    full_prompt = (
    f"Here is your message history with the user: Initialization: {initialization_prompt} {conversation.messages}\n\n"
    f"User now said: {message}.\n"
    f"Respond with a JSON object in the following format:\n\n"
    f"{{\n"
    f"  \"answer\": \"<Provide a thoughtful, empathetic response to the user's message here>\",\n"
    f"  \"therapy\": {{\n"
    f"    \"name\": \"<The name of the therapeutic approach used>\",\n"
    f"    \"reason\": \"<Explain why this technique was chosen based on the user's input>\",\n"
    f"    \"description\": \"<Provide a brief description of what this technique entails>\"\n"
    f"  }}\n"
    f"}}\n\n"
    f"Example:\n"
    f"{{\n"
    f"  \"answer\": \"Hey there. It sounds like you're looking to connect. How are you doing today?\",\n"
    f"  \"therapy\": {{\n"
    f"    \"name\": \"Person-Centered Therapy\",\n"
    f"    \"reason\": \"The user's initial greeting is neutral, indicating neither distress nor significant positive emotion. "
    f"Person-centered therapy's focus on empathy, unconditional positive regard, and genuineness provides a safe and non-judgmental "
    f"space for the user to open up at their own pace. This approach avoids imposing specific techniques prematurely and allows for "
    f"a natural unfolding of the therapeutic conversation.\",\n"
    f"    \"description\": \"Person-centered therapy emphasizes the inherent capacity for self-actualization within each individual. "
    f"The therapist provides a supportive and accepting environment, fostering the client's self-exploration and growth. "
    f"Key elements include empathy, unconditional positive regard, and congruence (genuineness) from the therapist.\"\n"
    f"  }}\n"
    f"}}\n\n"
    f"Ensure your response strictly follows this format and contains no additional text outside the JSON."
    )

    model = palm.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(full_prompt)

    # Parse the JSON response
    parsed_response = parse_model_response(response.text)

    # Update the conversation data
    try:
        conversation_data = json.loads(conversation.messages)
    except (json.JSONDecodeError, TypeError):
        # If messages are empty or invalid JSON, initialize as an empty list
        conversation_data = []
        app.logger.info("Initializing empty conversation data")

    # Append new messages
    conversation_data.append({"role": "user", "message": message, "emotion": emotion})
    conversation_data.append({
        "role": "ai",
        "answer": parsed_response.get("answer", ""),
        "therapy": parsed_response.get("therapy", {})
    })

    # Convert updated conversation data back to JSON string and save
    conversation.messages = json.dumps(conversation_data)
    db.session.commit()

    # Send back the updated conversation
    return jsonify({"response": parsed_response.get("answer", ""), "updated_messages": conversation_data})

def seed_admin_user():
    admin_username = 'admin'  
    admin_password = 'admin123'  
    
    # Check if an admin user already exists
    admin_user = User.query.filter_by(username=admin_username).first()
    if not admin_user:
        # Create the admin user
        admin_user = User(
            username=admin_username,
            password=admin_password,
            is_admin=True
        )
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created with username: admin and password: admin123")
    else:
        print("Admin user already exists.")


@app.route('/admin/tests', methods=['POST'])
@jwt_required()
def add_update_test():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    # Ensure only admin users can access this route
    if not user.is_admin:
        return jsonify({"message": "Admin access required"}), 403

    data = request.json
    test_id = data.get("id")
    name = data.get("name")
    description = data.get("description")
    format = data.get("format")

    # Validate input
    if not name or not format:
        return jsonify({"message": "Test name and format are required"}), 400

    # Update existing test or create a new one
    if test_id:
        test = Test.query.get(test_id)
        if not test:
            return jsonify({"message": "Test not found"}), 404
        test.name = name
        test.description = description
        test.format = format
    else:
        test = Test(name=name, description=description, format=format)
        db.session.add(test)

    db.session.commit()
    return jsonify({"message": "Test saved successfully"})

def seed_add_tests():
    """
    Adds multiple predefined tests to the database if they do not already exist.
    """
    tests = [
        {
            "name": "Borderline Symptom List 23",
            "description": (
                "The Borderline Symptom List 23 (BSL-23) is a self-assessment scale for recording the extent of intrapsychic stress in borderline patients. "
                "It is based on the criteria found in the DSM-IV, the Diagnostic Interview for Borderline Disorders (DIB-R), expert opinions and patient statements. "
                "It records the severity of the disorder and its changes over time."
            ),
            "format": {
                "questions": [
                    {
                        "text": "In the course of last week it was hard for me to concentrate",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },
                    {
                        "text": "In the course of last week I felt helpless",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },
                    {
                        "text": "In the course of last week I was absent-minded and unable to remember what I was actually doing",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },
                                          {
                        "text": "In the course of last week I felt disgust",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I thought of hurting myself",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I didn't trust other people",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I didn't believe in my right to live",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I was lonely",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I experienced stressful inner tension",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I had images that I was very much afraid of",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I hated myself",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I wanted to punish myself",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I suffered from shame",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week my mood rapidly cycled in terms of anxiety, anger, and depression",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I suffered from voices and noises from inside or outside my head",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week criticism had a devastating effect on me",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I felt vulnerable",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week the idea of death had a certain fascination for me",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week everything seemed senseless to me",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I was afraid of losing control",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },                    {
                        "text": "In the course of last week I felt disgusted by myself",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },
                                       {
                        "text": "In the course of last week I felt as if I was far away from myself",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    },
                    {
                        "text": "In the course of last week I felt worthless",
                        "options": [
                            {"text": "Not at all", "score": 0},
                            {"text": "A little", "score": 1},
                            {"text": "Rather", "score": 2},
                            {"text": "Much", "score": 3},
                            {"text": "Very strong", "score": 4}
                        ]
                    }
                ],
                "scoring_rule": "average",
            }
        },
        {
    "name": "Severity Measure for Agoraphobia—Adult",
    "description": (
        "The Severity Measure for Agoraphobia—Adult is a tool designed to evaluate thoughts, feelings, and behaviors related to agoraphobia in various situations. "
        "It is based on the rights granted by the American Psychiatric Association for use by researchers and clinicians without requiring prior permission."
    ),
    "format": {
        "questions": [
            {
                "text": "During the past 7 days, in these situations (crowds, public places, transportation, traveling alone, or away from home), I have felt moments of sudden terror, fear, or fright.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have felt anxious, worried, or nervous.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have had thoughts about panic attacks, uncomfortable physical sensations, getting lost, or being overcome with fear.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have felt a racing heart, sweaty, trouble breathing, faint, or shaky.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have felt tense muscles, felt on edge or restless, or had trouble relaxing.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have avoided, or did not approach or enter them.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have moved away from them, left them early, or remained close to the exits.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have spent a lot of time preparing for, or procrastinating about them.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have distracted myself to avoid thinking about them.",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            },
            {
                "text": "During the past 7 days, in these situations, I have needed help to cope with them (e.g., alcohol or medication, superstitious objects, other people).",
                "options": [
                    {"text": "Never", "score": 0},
                    {"text": "Occasionally", "score": 1},
                    {"text": "Half of the time", "score": 2},
                    {"text": "Most of the time", "score": 3},
                    {"text": "All of the time", "score": 4}
                ]
            }
        ],
        "scoring_rule": "average"
    }
},
{
    "name": "Chronic Pain Acceptance Questionnaire – Revised (CPAQ-R)",
    "description": "The Chronic Pain Acceptance Questionnaire – Revised (CPAQ-R) is a 20-item tool designed to measure the acceptance of chronic pain. Acceptance of pain reduces unsuccessful attempts to avoid or control pain, allowing individuals to focus on engaging in valued activities and pursuing meaningful goals. The questionnaire evaluates two factors:\n1. Activity engagement: Pursuit of life activities regardless of pain (Items: 1, 2, 3, 5, 6, 8, 9, 10, 12, 15, 19).\n2. Pain willingness: Recognition that avoidance and control are often unworkable methods of adapting to chronic pain (Items: 4, 7, 11, 13, 14, 16, 17, 18, 20).\nScoring: Each item is rated on a 7-point scale (0 = Never true, 6 = Always true). Scores for both factors are summed, and the total score reflects higher levels of pain acceptance.",
    "format": {
        "questions": [
            {
                "text": "I am getting on with the business of living no matter what my level of pain is.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "My life is going well, even though I have chronic pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "It’s OK to experience pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "I would gladly sacrifice important things in my life to control this pain better.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "It’s not necessary for me to control my pain in order to handle my life well.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "Although things have changed, I am living a normal life despite my chronic pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "I need to concentrate on getting rid of my pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "There are many activities I do when I feel pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "I lead a full life even though I have chronic pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "Controlling my pain is less important than any other goals in my life.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "My thoughts and feelings about pain must change before I can take important steps in my life.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "Despite the pain, I am now sticking to a certain course in my life.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "Keeping my pain level under control takes first priority whenever I’m doing something.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "Before I can make any serious plans, I have to get some control over my pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "When my pain increases, I can still take care of my responsibilities.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "I will have better control over my life if I can control my negative thoughts about pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "I avoid putting myself in situations where my pain might increase.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "My worries and fears about what pain will do to me are true.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
            {
                "text": "It’s a great relief to realize that I don’t have to change my pain to get on with life.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            },
                        {
                "text": "I have to struggle to do things when I have pain.",
                "options": [
                    {"text": "Never true", "score": 0},
                    {"text": "Very rarely true", "score": 1},
                    {"text": "Seldom true", "score": 2},
                    {"text": "Sometimes true", "score": 3},
                    {"text": "Often true", "score": 4},
                    {"text": "Almost always true", "score": 5},
                    {"text": "Always true", "score": 6}
                ]
            }
        ],
        "scoring_rule": "sum"
    }
},
{
    "name": "The CAT-Q",
    "description": (
        '''The CAT-Q is a sub-test designed to measure the degree to which you use camouflaging strategies in social interactions, particularly in the context of autism. Camouflaging refers to the suppression of autistic traits to adapt to social norms, which can be both conscious and unconscious. The more you camouflage, the more of your autistic traits you may suppress. A high camouflaging score may also explain why you don’t meet diagnostic criteria for autism, yet still experience autistic traits. This tool helps to identify the extent of camouflaging you engage in, which could be masking underlying autistic behaviors.

Users can also manually calculate subtraits with this:

Self-scoring:  
If auto-scoring isn’t working, or if you prefer to self-score your answers, follow the steps below.

CAT-Q total score:  
Reverse the scores of the answers for statements 3, 12, 19, 22, and 24.  
Original scoring: a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7  
Reverse scoring: a = 7, b = 6, c = 5, d = 4, e = 3, f = 2, g = 1  
Then, add up all answers for statements 1–25.

Compensation score:  
Add up all answers for statements 1, 4, 5, 8, 11, 14, 17, 20, and 23.

Masking score:  
Add up all answers for statements 2, 6, 9, 12, 15, 18, 21, and 24.  
Use the reversed scoring for statements 12 and 24.

Assimilation score:  
Add up all answers for statements 3, 7, 10, 13, 16, 19, 22, and 25.  
Use the reversed scoring for statements 3, 19, and 22.

Average scores:  
A total score of 100 or above indicates you camouflage autistic traits, meaning you suppress certain behaviors to fit into social contexts.
'''
    ),
    "format": {
        "questions": [
            {
                "text": "When I am interacting with someone, I deliberately copy their body language or facial expressions.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I monitor my body language or facial expressions so that I appear relaxed.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I rarely feel the need to put on an act in order to get through a social situation.",
                "options": [
                    {"text": "Strongly Disagree", "score": 6},
                    {"text": "Disagree", "score": 5},
                    {"text": "Somewhat Disagree", "score": 4},
                    {"text": "Neither Agree nor Disagree", "score": 3},
                    {"text": "Somewhat Agree", "score": 2},
                    {"text": "Agree", "score": 1}
                ]
            },
            {
                "text": "I have developed a script to follow in social situations.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I will repeat phrases that I have heard others say in the exact same way that I first heard them.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I adjust my body language or facial expressions so that I appear interested by the person I am interacting with.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "In social situations, I feel like I’m ‘performing’ rather than being myself.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "In my own social interactions, I use behaviours that I have learned from watching other people interacting.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I always think about the impression I make on other people.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I need the support of other people in order to socialise.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I practice my facial expressions and body language to make sure they look natural.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I don’t feel the need to make eye contact with other people if I don’t want to.",
                "options": [
                    {"text": "Strongly Disagree", "score": 6},
                    {"text": "Disagree", "score": 5},
                    {"text": "Somewhat Disagree", "score": 4},
                    {"text": "Neither Agree nor Disagree", "score": 3},
                    {"text": "Somewhat Agree", "score": 2},
                    {"text": "Agree", "score": 1}
                ]
            },
            {
                "text": "I have to force myself to interact with people when I am in social situations.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I have tried to improve my understanding of social skills by watching other people.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I monitor my body language or facial expressions so that I appear interested by the person I am interacting with.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "When in social situations, I try to find ways to avoid interacting with others.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I have researched the rules of social interactions to improve my own social skills.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I am always aware of the impression I make on other people.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I feel free to be myself when I am with other people.",
                "options": [
                    {"text": "Strongly Disagree", "score": 6},
                    {"text": "Disagree", "score": 5},
                    {"text": "Somewhat Disagree", "score": 4},
                    {"text": "Neither Agree nor Disagree", "score": 3},
                    {"text": "Somewhat Agree", "score": 2},
                    {"text": "Agree", "score": 1}
                ]
            },
            {
                "text": "I learn how people use their bodies and faces to interact by watching television or films, or by reading fiction.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "I adjust my body language or facial expressions so that I appear relaxed.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "When talking to other people, I feel like the conversation flows naturally.",
                "options": [
                    {"text": "Strongly Disagree", "score": 6},
                    {"text": "Disagree", "score": 5},
                    {"text": "Somewhat Disagree", "score": 4},
                    {"text": "Neither Agree nor Disagree", "score": 3},
                    {"text": "Somewhat Agree", "score": 2},
                    {"text": "Agree", "score": 1}
                ]
            },
            {
                "text": "I have spent time learning social skills from television shows and films, and try to use these in my interactions.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            },
            {
                "text": "In social interactions, I do not pay attention to what my face or body are doing.",
                "options": [
                    {"text": "Strongly Disagree", "score": 6},
                    {"text": "Disagree", "score": 5},
                    {"text": "Somewhat Disagree", "score": 4},
                    {"text": "Neither Agree nor Disagree", "score": 3},
                    {"text": "Somewhat Agree", "score": 2},
                    {"text": "Agree", "score": 1}
                ]
            },
            {
                "text": "In social situations, I feel like I am pretending to be ‘normal’.",
                "options": [
                    {"text": "Strongly Disagree", "score": 1},
                    {"text": "Disagree", "score": 2},
                    {"text": "Somewhat Disagree", "score": 3},
                    {"text": "Neither Agree nor Disagree", "score": 4},
                    {"text": "Somewhat Agree", "score": 5},
                    {"text": "Agree", "score": 6}
                ]
            }
        ],
        "scoring_rule": "sum",
    }
},
{
            "name": "The Leeds Dependance",
            "description": (
                "A concise self-assessment questionnaire consisting of 10 items, designed to evaluate the severity of dependence on any drug, including alcohol. "
                "Respondents are asked to reflect on the past week and focus on their primary substance(s) of use, specifying the substance(s) when applicable."
                "Each question offers four response options: Never (0), Sometimes (1), Often (2), and Nearly Always (3), allowing individuals to select the most appropriate answer."
            ),
            "format": {
                "questions": [
                      {
                          "text": "Do you find yourself thinking about when you will next be able to have another drink or take drugs?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Is drinking or taking drugs more important than anything else you might do during the day?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you feel your need for drink or drugs is too strong to control?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you plan your days around getting and taking drink or drugs?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you drink or take drugs in a particular way in order to increase the effect it gives you?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you drink or take drugs morning, afternoon and evening?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you feel you have to carry on drinking or taking drugs once you have started?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Is getting the effect you want more important than the particular drink or drug you use?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you want to take more drink or drugs when the effect starts to wear off?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      },
                      {
                          "text": "Do you find it difficult to cope with life without drink or drugs?",
                          "options": [
                              {"text": "Never", "score": 0},
                              {"text": "Sometimes", "score": 1},
                              {"text": "Often", "score": 2},
                              {"text": "Nearly always", "score": 3}
                          ]
                      }
                ],
                "scoring_rule": "sum",
            }
},
{
            "name": "Brief Fear of Negative Evaluation Scale",
            "description": (
                "The Brief Fear of Negative Evaluation Scale (Leary, 1983) is a self-assessment tool designed to measure an individual's concern about being negatively evaluated by others."
                "Participants are presented with a series of statements and asked to rate how characteristic each statement is of them using the following scale: 1: Not at all characteristic of me, 2: Slightly characteristic of me, 3: Moderately characteristic of me, 4: Very characteristic of me, 5: Extremely characteristic of me"
                "The scale provides insights into the participant's levels of social anxiety and fear of judgment, aiding in the evaluation of their emotional and interpersonal dynamics."
            ),
            "format": {
                "questions": [
                  {
                      "text": "I worry about what other people will think of me even when I know it doesn't make any difference.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I am unconcerned even if I know people are forming an unfavorable impression of me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I am frequently afraid of other people noticing my shortcomings.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I rarely worry about what kind of impression I am making on someone.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I am afraid others will not approve of me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I am afraid that people will find fault with me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "Other people's opinions of me do not bother me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "When I am talking to someone, I worry about what they may be thinking about me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I am usually worried about what kind of impression I make.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "If I know someone is judging me, it has little effect on me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "Sometimes I think I am too concerned with what other people think of me.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  },
                  {
                      "text": "I often worry that I will say or do the wrong things.",
                      "options": [
                          {"text": "Not at all characteristic of me", "score": 1},
                          {"text": "Slightly characteristic of me", "score": 2},
                          {"text": "Moderately characteristic of me", "score": 3},
                          {"text": "Very characteristic of me", "score": 4},
                          {"text": "Extremely characteristic of me", "score": 5}
                      ]
                  }
                ],
                "scoring_rule": "sum",
            }
        },
        {
    "name": "CUDOS: Depression Scale",
    "description": (
        "The CUDOS Depression Scale is a self-assessment tool designed to measure symptoms of depression. "
        "Participants are presented with a series of statements and asked to rate how well each statement describes their experiences "
        "during the past week, including today, using the following scale: "
        "0 = Not at all true (0 days), 1 = Rarely true (1-2 days), 2 = Sometimes true (3-4 days), "
        "3 = Often true (5-6 days), 4 = Almost always true (every day). "
        "This scale provides insights into the participant's mood, behavior, and emotional state, aiding in the evaluation of depressive symptoms."
    ),
    "format": {
        "questions": [
            {
                "text": "I felt sad or depressed.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I was not as interested in my usual activities.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "My appetite was poor and I didn't feel like eating.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "My appetite was much greater than usual.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I had difficulty sleeping.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I was sleeping too much.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I felt very fidgety, making it difficult to sit still.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I felt physically slowed down, like my body was stuck in mud.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "My energy level was low.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I felt guilty.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I thought I was a failure.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I had problems concentrating.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I had more difficulties making decisions than usual.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I thought about killing myself.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": ". I wished I was dead.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
            {
                "text": "I thought that the future looked hopeless.",
                "options": [
                    {"text": "Not at all true (0 days)", "score": 0},
                    {"text": "Rarely true (1-2 days)", "score": 1},
                    {"text": "Sometimes true (3-4 days)", "score": 2},
                    {"text": "Often true (5-6 days)", "score": 3},
                    {"text": "Almost always true (every day)", "score": 4}
                ]
            },
        ],
        "scoring_rule": "sum"
    }
},
{
    "name": "Interpersonal Needs Questionnaire (INQ)",
    "description": (
        "The Interpersonal Needs Questionnaire (INQ) is a self-assessment tool designed to measure two core constructs "
        "related to suicidal ideation: thwarted belongingness and perceived burdensomeness. Participants are asked to "
        "rate how true each statement is based on how they have been feeling recently, using a 7-point scale ranging from "
        "1 = Not at all true for me to 7 = Very true for me. Reverse coding is applied to items 7, 8, 10, 13, 14, and 15."
    ),
    "format": {
        "questions": [
            {
                "text": "These days, the people in my life would be better off if I were gone.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, the people in my life would be happier without me.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I think I am a burden on society.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I think my death would be a relief to the people in my life.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I think the people in my life wish they could be rid of me.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I think I make things worse for the people in my life.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, other people care about me.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            },
            {
                "text": "These days, I feel like I belong.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            },
            {
                "text": "These days, I rarely interact with people who care about me.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I am fortunate to have many caring and supportive friends.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            },
            {
                "text": "These days, I feel disconnected from other people.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I often feel like an outsider in social gatherings.",
                "options": [
                    {"text": "Not at all true for me", "score": 1},
                    {"text": "Somewhat true for me", "score": 2},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Very true for me", "score": 6},
                    {"text": "Very true for me", "score": 7}
                ]
            },
            {
                "text": "These days, I feel that there are people I can turn to in times of need.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            },
            {
                "text": "These days, I am close to other people.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            },
            {
                "text": "These days, I have at least one satisfying interaction every day.",
                "options": [
                    {"text": "Not at all true for me", "score": 7},
                    {"text": "Somewhat true for me", "score": 6},
                    {"text": "Very true for me", "score": 5},
                    {"text": "Somewhat true for me", "score": 4},
                    {"text": "Very true for me", "score": 3},
                    {"text": "Very true for me", "score": 2},
                    {"text": "Very true for me", "score": 1}
                ]
            }
        ],
        "scoring_rule": "sum"
    }
},
{
            "name": "Buss & Perry Aggression",
            "description": (
                "The Aggression Questionnaire (Buss & Perry, 1992) is a self-assessment tool designed to evaluate levels of aggression across various dimensions. "
                "Participants rate how well each statement describes them using a 5-point scale: 1 (extremely uncharacteristic of me), 2 (somewhat uncharacteristic of me), 3 (neither uncharacteristic nor characteristic of me), 4 (somewhat characteristic of me), and 5 (extremely characteristic of me)."
                "Ratings are recorded next to each statement to reflect the respondent's self-perception."
            ),
            "format": {
                "questions": [
                        {
                            "text": "Some of my friends think I am a hothead.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "If I have to resort to violence to protect my rights, I will.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "When people are especially nice to me, I wonder what they want.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I tell my friends openly when I disagree with them.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I have become so mad that I have broken things.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I can’t help getting into arguments when people disagree with me.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I wonder why sometimes I feel so bitter about things.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "Once in a while, I can’t control the urge to strike another person.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I am an even-tempered person.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 5},
                                {"text": "Somewhat uncharacteristic of me", "score": 4},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 2},
                                {"text": "Extremely characteristic of me", "score": 1}
                            ]
                        },
                        {
                            "text": "I am suspicious of overly friendly strangers.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I have threatened people I know.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I flare up quickly but get over it quickly.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "Given enough provocation, I may hit another person.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "When people annoy me, I may tell them what I think of them.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I am sometimes eaten up with jealousy.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I can think of no good reason for ever hitting a person.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 5},
                                {"text": "Somewhat uncharacteristic of me", "score": 4},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 2},
                                {"text": "Extremely characteristic of me", "score": 1}
                            ]
                        },
                        {
                            "text": "At times I feel I have gotten a raw deal out of life.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I have trouble controlling my temper.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "When frustrated, I let my irritation show.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I sometimes feel that people are laughing at me behind my back.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I often find myself disagreeing with people.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "If somebody hits me, I hit back.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I sometimes feel like a powder keg ready to explode.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "Other people always seem to get the breaks.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "There are people who pushed me so far that we came to blows.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I know that “friends” talk about me behind my back.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "My friends say that I’m somewhat argumentative.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "Sometimes I fly off the handle for no good reason.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        },
                        {
                            "text": "I get into fights a little more than the average person.",
                            "options": [
                                {"text": "Extremely uncharacteristic of me", "score": 1},
                                {"text": "Somewhat uncharacteristic of me", "score": 2},
                                {"text": "Neither uncharacteristic nor characteristic of me", "score": 3},
                                {"text": "Somewhat characteristic of me", "score": 4},
                                {"text": "Extremely characteristic of me", "score": 5}
                            ]
                        }

                ],
                "scoring_rule": "sum",
            }
        }
]

    for test in tests:
        existing_test = Test.query.filter_by(name=test["name"]).first()
        if existing_test:
            print(f"Test '{test['name']}' already exists in the database.")
            continue

        new_test = Test(name=test["name"], description=test["description"], format=test["format"])
        db.session.add(new_test)
        print(f"Test '{test['name']}' has been added to the database.")

    db.session.commit()
    print("All tests have been seeded.")


@app.route('/api/my-tests', methods=['GET'])
@jwt_required()  # Ensure the user is authenticated
def get_user_tests():
    user_id = get_jwt_identity()  # Get user ID from the JWT token
    user_tests = UserTest.query.filter_by(user_id=user_id).all()
    
    # Serialize the results
    results = [
        {
            "test_id": user_test.test.id,
            "test_name": user_test.test.name,
            "test_description": user_test.test.description,
            "score": user_test.score
        } 
        for user_test in user_tests
    ]
    return jsonify(results), 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_admin_user()
        seed_add_tests()
    app.run(debug=True)
