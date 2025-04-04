# Backend TrueLens

This is the backend of the **TrueLens** project, a Flask-based fake news detection system using machine learning.

## 🚀 Features
- Fake news detection from text input
- Optical Character Recognition (OCR) to extract text from images
- REST API with Flask
- Model training using Scikit-learn
- Cross-Origin Resource Sharing (CORS) enabled for frontend communication

## 🛠 Tech Stack
- **Backend:** Flask, Flask-CORS
- **Machine Learning:** Scikit-learn, NeatText
- **OCR:** Pytesseract
- **Database:** CSV-based dataset (replaceable with SQL/NoSQL in future)

## 📦 Setup & Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/amit-yadav207/backend_TrueLens.git
cd backend_TrueLens
```


### 2️⃣ Create and Activate Virtual Environment
```sh
python -m venv myenv
myenv\Scripts\activate
```


### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```sh
python app.py
```

### 📤 API Endpoints
**Method**	**Endpoint**	**Description**
- POST	/predict_text	Predicts if the input text is real or fake
- POST	/predict_image	Extracts text from an image and predicts if it's real or fake