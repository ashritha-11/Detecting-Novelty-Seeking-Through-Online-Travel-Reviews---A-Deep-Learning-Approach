# 🧭 Novelty-Seeking Recognition & Travel Recommendation

## 📌 Project Overview
This project is a deep-learning-powered web application designed to identify **Novelty-Seeking** personality traits from travel reviews and user prompts. By leveraging a hybrid **BERT + BiGRU** architecture, the system classifies travel intent into four psychological profiles, helping users discover destinations that match their current state of mind.

## ✨ Key Features
- **Intelligent Intent Detection**: Uses a fine-tuned BERT-BiGRU model to analyze user text and classify it into core novelty profiles.
- **Dynamic Glassmorphism UI**: A premium, modern interface with high-end visual effects and responsive design.
- **Cinematic Backgrounds**: Logged-in users experience a dynamic background that cycles through 12 high-definition travel images with smooth transitions and breathing animations.
- **Interactive Trends Dashboard**: Real-time visualization of dataset distribution, user ratings, and deep learning model performance metrics (Accuracy, F1 Score) using Chart.js.
- **Secure Authentication Flow**: Enforced registration and login system to protect personalized experiences.
- **Smart Data Access**: Integrated geolocation mapping and keyword-based image relevance for an immersive exploration of global attractions.

## 🧠 Model Architecture
The core classification engine uses a state-of-the-art Natural Language Processing (NLP) pipeline:
1.  **BERT (Bidirectional Encoder Representations from Transformers)**: Extracts deep semantic features from text.
2.  **BiGRU (Bidirectional Gated Recurrent Unit)**: Captures sequential dependencies and context in travel reviews.
3.  **Softmax Classifier**: Outputs probabilities for the four novelty types:
    -   **Experience**: Seeking cultural and historical immersion.
    -   **Arousal**: Looking for thrills and high-energy adventure.
    -   **Relaxation**: Quiet, peaceful places to unwind.
    -   **Boredom Alleviation**: Seeking a variety-rich escape from routine.

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (ES6+)
- **Visualizations**: Chart.js
- **Database**: SQLite (Auth), CSV (Travel Data)
- **Maps**: Leaflet.js (OpenStreetMap)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed and a virtual environment active.

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd "Detecting novelty seeking-Mini project"

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
The application is configured to run on **port 5010** to avoid common port conflicts.
```bash
python web/app.py
```
Visit `http://127.0.0.1:5010/register` in your browser to begin.

## 📊 Project Structure
- `web/`: Contains the Flask application, templates, and static assets (CSS/JS/Images).
- `src/`: Core Python logic including model definitions, prediction scripts, and data access layers.
- `models/`: Stores the pre-trained `.pt` model weights.
- `data/`: Processed TripAdvisor datasets and geocaching files.

---
*Developed as a Mini Project for detecting psychological travel patterns through advanced NLP.*
