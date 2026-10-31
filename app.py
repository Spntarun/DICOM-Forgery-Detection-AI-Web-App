import os
import io
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image

import model_inference

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'srm_dicom_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize Database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Failed. Please check your email and password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Handle case where user uploads same filename over and over by adding timestamp
        base, ext = os.path.splitext(filename)
        unique_name = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)
        
        try:
            # 1. Process Image & Extract Metadata
            pil_raw = Image.open(filepath)
            
            metadata = {
                "Format": pil_raw.format if pil_raw.format else "Unknown",
                "Color Mode": pil_raw.mode,
                "Dimensions": f"{pil_raw.size[0]} x {pil_raw.size[1]} px"
            }
            
            # Attempt to pull basic EXIF
            exif = pil_raw.getexif()
            if exif:
                from PIL.ExifTags import TAGS
                for tag_id in exif:
                    tag = TAGS.get(tag_id, tag_id)
                    val = exif.get(tag_id)
                    if isinstance(val, bytes):
                        try:
                            val = val.decode()
                        except:
                            val = "<Binary Data>"
                    if isinstance(tag, str) and len(str(val)) < 150:
                        metadata[f"EXIF: {tag}"] = str(val)

            pil_img = pil_raw.convert('RGB')
            
            # 2. Extract visualization maps
            # ELA Map
            ela_map = model_inference.compute_ela(pil_img, quality=85, scale=25)
            ela_filename = f"ela_{unique_name}"
            ela_path = os.path.join(app.config['PROCESSED_FOLDER'], ela_filename)
            ela_map.save(ela_path)
            
            # Noise Map (Using SRM-like Laplacian approximation for visualization)
            noise_map = model_inference.compute_noise_map_for_viz(pil_img)
            noise_filename = f"noise_{unique_name}"
            noise_path = os.path.join(app.config['PROCESSED_FOLDER'], noise_filename)
            noise_map.save(noise_path)
            
            # 3. Model Inference
            is_forged, class_name, conf = model_inference.predict_single_image(pil_img)
            
            return jsonify({
                'success': True,
                'class_name': class_name,
                'is_forged': is_forged,
                'confidence': f"{conf:.2f}",
                'metadata': metadata,
                'original_url': url_for('static', filename=f"uploads/{unique_name}"),
                'ela_url': url_for('static', filename=f"processed/{ela_filename}"),
                'noise_url': url_for('static', filename=f"processed/{noise_filename}")
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the model weights loading before web requests start
    model_inference.load_model()
    # Run server
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
