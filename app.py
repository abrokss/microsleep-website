from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("best.pt")  # path ke model

camera = cv2.VideoCapture(0)  # gunakan 0 untuk webcam laptop

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Deteksi dengan YOLO
            results = model(frame)
            annotated_frame = results[0].plot()

            # Encode ke JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Kirim frame via HTTP stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
