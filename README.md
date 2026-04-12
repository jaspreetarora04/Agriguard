# 🌱 AgriGuard

## Smart Infection Detection & Intelligent Pesticide Spraying System

AgriGuard is an AI-powered smart agriculture system that detects plant diseases in real-time and performs **precision pesticide spraying based on infection level**, reducing chemical wastage and improving crop health.

---

## 📌 Problem Statement

Traditional farming practices rely on **uniform pesticide spraying**, which leads to:

*  Excessive chemical usage (~60% waste)
*  Soil degradation & water pollution
*  Manual inspection (time-consuming & inaccurate)
*  Spraying on healthy plants (unnecessary cost)

There is a need for an **automated, intelligent, and cost-effective system** that can detect plant infection and apply pesticides only where required.

---

## 💡 Proposed Solution

AgriGuard uses **Computer Vision + IoT** to create a smart spraying system that:

* Detects plant infection using a **CNN model**
* Classifies infection levels (**Healthy / Mild / Severe**)
* Automatically sprays pesticide **based on severity**
* Avoids spraying on healthy plants

---

## ⚙️ Key Features

* 🤖 **AI-Based Disease Detection (CNN)**
* 🎯 **Targeted Pesticide Spraying**
* 📊 **Infection-Level Based Decision System**
* 💧 **Automatic Pump Control (MOSFET switching)**
* 📷 **ESP32-CAM Image Capture**
* 🌐 **Flask Web Interface for Monitoring**
* ♻️ **Eco-Friendly & Cost-Effective**

---

## 🧠 System Architecture

![Architecture](docs/architecture.png)

---

## ⚙️ Hardware Components

* ESP32-CAM (Image Capture + Processing Unit)
* N-channel MOSFET (IRLZ44N) – Pump switching
* 12V Diaphragm Water Pump
* Nozzle (Spray mechanism)
* Power Supply (5V & 12V)
* Connecting wires & circuit

---

## 💻 Software Stack

* Python 3
* Flask (Backend Server)
* Custom CNN Model

---

## 🔄 Working Flow

### Step 1: Image Capture

ESP32-CAM captures real-time image of plant leaves.

### Step 2: AI Analysis

Image is sent to Flask server → processed using CNN model.

### Step 3: Prediction Output

Model classifies plant into:

* ✅ Healthy
* ⚠️ Mild Infection
* ❌ Severe Infection

### Step 4: Decision Logic

* Healthy → No spray
* Mild → Short spray (e.g., 1 sec)
* Severe → Longer spray (e.g., 3 sec)

### Step 5: Action

ESP32 triggers MOSFET → pump ON → pesticide sprayed.

---

## ⚙️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Agriguard.git
cd Agriguard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask Server

```bash
python backend/app.py
```

### 4. Setup Hardware

* Upload code to ESP32-CAM
* Connect MOSFET + Pump
* Ensure same WiFi network

### 5. Start System

* Capture image
* AI detects infection
* Pump automatically sprays

---

## 📊 Results

![Output](results/sample_output.png)

* 🎯 Accurate disease detection
* 💧 Reduced pesticide usage (~70%)
* ⚡ Real-time automated response

---

## 🎥 Demo

[Watch Demo](demo/video_link.txt)

---

## 🧪 Innovation

Unlike traditional systems, AgriGuard introduces:

* **AI-driven infection-level based spraying**
* **Precision agriculture at low cost**
* **Real-time decision + action system**
* **Hardware + AI integration in a single pipeline**

---

## 🌍 Impact

* Reduces environmental pollution
* Saves pesticide cost
* Improves crop yield
* Supports small-scale farmers

---

## 🔮 Future Scope

* Multi-disease classification
* Mobile app integration
* Cloud-based analytics dashboard
* Solar-powered system
* Weather-based spraying logic

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Authors

* Jaspreet
* Vartika Singh
* Divyanshi Katiyar
* Saksham Singhal


B.Tech – Electronics & Communication Engineering

---
