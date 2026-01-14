AI Human Safety System for Industrial Use

A real-time computer vision–based safety monitoring system designed to detect human presence in restricted industrial zones and trigger alerts to prevent workplace accidents.
🚀 Key Features

Real-time human detection using YOLOv8

Restricted zone violation detection using bounding-box center logic

Audio and visual alerts for immediate response

Event logging with timestamps for audit and compliance

Modular and configuration-driven design

🛠 Tech Stack

Language: Python

Computer Vision: OpenCV

Deep Learning Model: YOLOv8 (Ultralytics)

📂 Project Structure

AI-Human-Safety-System/
│── src/
│   └── main.py
│── model/
│   └── yolov8n.pt
│── config.json
│── README.md
│── .gitignore


▶️ How to Run
git clone https://github.com/hasanpatel345/AI-Human-Safety-System.git
cd AI-Human-Safety-System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd src
python main.py


Press Q to stop the camera.

🔮 Future Enhancements

Multi-camera support

Cloud-based event storage

Integration with industrial IoT systems
