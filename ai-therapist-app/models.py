# Database models for user data
from flask_sqlalchemy import SQLAlchemy

# Initialize the database
db = SQLAlchemy()

class UserHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    response = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
