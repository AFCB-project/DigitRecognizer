# DigitRecognizer
This is OpenSource AI where you put your image and AI learns to understand numbers


🧠 **A neural network built from scratch in C# to recognize digits (0–9) from images.**  
💡 Learning is interactive: the user provides feedback (correct or incorrect), and the model learns on the fly.

---

## 🚀 How It Works

1. Place an image named `Pic.png` into the `Data/` folder  
2. Run the program (`Program.cs`)  
3. The neural network will output a guess:
I think this is: 6
Is that correct? (1 = yes, 0 = no):

yaml
Копировать
Редактировать
4. If correct — input `1`  
If incorrect — input `0`, then provide the correct label (e.g., `3`)  
5. The image is saved as `Datasource/DataN.png` and used to train the model

---

## 📁 Project Structure

DigitRecognizer/
├── AI/
│ ├── NeuralNetwork.cs # Neural network architecture, training, prediction
│ └── ImageProcessor.cs # Image loading and preprocessing
├── Data/
│ └── Pic.png # Input image to be recognized
├── Datasource/
│ └── Data1.png, Data2.png # Stored labeled images for training
├── Program.cs # Main loop: prediction and manual training
└── Readme.md # This file

yaml
Копировать
Редактировать

---

## ✅ Requirements

- .NET 6.0 or later
- Uses `System.Drawing`  
  *(On Linux, you may need to install `libgdiplus` for image processing)*

---

## 🧩 Features

- Trains from scratch (no external libraries or datasets)
- Supports `.png` and `.jpeg` image formats
- Stores labeled training data automatically
- Learns from your feedback in real time

---

## 🔮 Future Plans

- GUI interface
- Accuracy evaluation/testing
- Support for other image sizes and formats

---

## 📸 Image Guidelines

- Size: **28x28 pixels**
- Format: **Black and white** (white digit on black background preferred)

---

Enjoy teaching your neural net — one image at a time! 😄
