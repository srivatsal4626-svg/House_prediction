import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

# -------------------- APP CONFIG --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey_change_in_production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///house_price.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# -------------------- LOAD MODEL --------------------
model = None
encoder = None

try:
    if os.path.exists('model.pkl') and os.path.exists('encoder.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    else:
        print("Model or encoder file not found.")
except Exception as e:
    print("Error loading model:", e)

# -------------------- USER MODEL --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- ROUTES --------------------

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# -------------------- REGISTER --------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            flash("All fields are required", "danger")
            return redirect(url_for('register'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('login'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(name=name, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully. You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# -------------------- LOGIN --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

# -------------------- LOGOUT --------------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

# -------------------- DASHBOARD --------------------
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None

    if request.method == 'POST':
        try:
            if model is None or encoder is None:
                flash("Model not loaded. Please check model files.", "danger")
                return redirect(url_for('dashboard'))

            area = float(request.form.get('area') or 0)
            bedrooms = int(request.form.get('bedrooms') or 0)
            bathrooms = int(request.form.get('bathrooms') or 0)
            location = request.form.get('location')

            # Validate location
            if location not in encoder.classes_:
                flash("Invalid location selected", "danger")
                return redirect(url_for('dashboard'))

            location_encoded = encoder.transform([location])[0]

            features = np.array([[area, bedrooms, bathrooms, location_encoded]])

            predicted_price = model.predict(features)[0]
            prediction = round(predicted_price, 2)

        except Exception as e:
            print("Prediction Error:", e)
            flash("Error in prediction. Please check inputs.", "danger")

    locations = encoder.classes_ if encoder else []
    return render_template('dashboard.html', prediction=prediction, locations=locations)

# -------------------- RUN APP --------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
