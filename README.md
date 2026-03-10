# 🍓 FruitSense — AI Ripeness Detector

A Flask web application that uses a Teachable Machine image classification model
to detect whether fruit is **ripe** or **unripe**.

---

## Project Structure

```
fruit-ripeness-detector/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── README.md
├── model/
│   ├── keras_model.h5      ← ⚠️  Add this after training (see below)
│   └── labels.txt          ← ⚠️  Add this after training (see below)
└── templates/
    └── index.html          ← Frontend webpage
```

---

## Step 1 — Train your model

1. Go to [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com)
2. Click **Get Started → Image Project → Standard image model**
3. Create classes like `Ripe Banana`, `Unripe Banana`, etc.
4. Upload your images and click **Train Model**
5. Click **Export Model → TensorFlow → Keras → Download**
6. Unzip the download and copy `keras_model.h5` and `labels.txt` into the `/model` folder

---

## Step 2 — Install dependencies

Make sure Python 3.9+ is installed, then run:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the app

```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## How it works

1. User uploads a fruit photo on the webpage
2. The image is sent to the Flask `/predict` route
3. Flask preprocesses the image (resizes to 224×224, normalises pixel values)
4. The Keras model runs inference and returns probabilities for each class
5. The result (label + confidence score + tip) is sent back as JSON
6. JavaScript displays the result with an animated confidence bar

---

## Tech Stack

- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Backend**: Python, Flask
- **AI Model**: TensorFlow / Keras (trained with Google Teachable Machine)
