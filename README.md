# 🩺 Pneumonia Detection System using Flask & PyTorch

A deep learning-based web application that detects **Pneumonia from Chest X-ray images** using a trained **ResNet18 model** built with **PyTorch** and deployed through **Flask**.

The system allows users to upload chest X-ray images and get instant predictions through an interactive web interface.

---

## 🚀 Features

- Upload Chest X-ray images
- Detect Pneumonia using Deep Learning
- Built with PyTorch and Flask
- Interactive and simple UI
- Fast prediction system
- Deployable on Render

---

## 🛠️ Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```bash
pneumonia-detection-flask/
│
├── model/
├── static/
├── templates/
├── app.py
├── pneumonia_resnet18.pth
├── requirements.txt
└── README.md
```

---

## 🧠 Model Used

The project uses a **ResNet18 CNN architecture** trained on Chest X-ray datasets for binary classification:

- Normal
- Pneumonia

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Rohitdubey-tech/pneumonia-detection-flask.git
```

### Navigate to Project Folder

```bash
cd pneumonia-detection-flask
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```bash
http://127.0.0.1:5000
```

---

## 📸 How It Works

1. Upload a Chest X-ray image
2. Model processes the image
3. Prediction is generated
4. Result is displayed on the webpage

---

## 🌐 Deployment

This project is deployed using:

- Render
- GitHub

---

## 📦 Requirements

Some major libraries used:

```txt
Flask
torch
torchvision
numpy
Pillow
gunicorn
```

---

## 🔮 Future Improvements

- Improve model accuracy
- Add confidence score
- Support multiple diseases
- Add user authentication
- Deploy mobile-friendly UI

---

## 👨‍💻 Author

**Rohit Kumar Dubey**

- GitHub: https://github.com/Rohitdubey-tech

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.
