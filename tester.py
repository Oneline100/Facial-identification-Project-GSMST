
import cv2
import time
import pickle
import tkinter as tk
import face_recognition
import imutils
from threading import Thread
from imutils.video import VideoStream, FPS

# Constants
ENCODINGS_PATH = "encodings.pickle"
CONFIDENCE_INTERVAL = 1
VIDEO_STREAM_SRC = 0
VIDEO_STREAM_FRAMERATE = 10

# Global variables
current_name = "unknown"
running = True

def load_data():
    print("[INFO] loading encodings + face detector...")
    return pickle.loads(open(ENCODINGS_PATH, "rb").read())

def start_video_stream():
    vs = VideoStream(src=VIDEO_STREAM_SRC, framerate=VIDEO_STREAM_FRAMERATE).start()
    time.sleep(2.0)
    return vs

def start_fps_counter():
    fps = FPS().start()
    return fps

def calculate_confidence_rate(encoding, data):
    min_distance = min(face_recognition.face_distance(data["encodings"], encoding))
    return (1 - min_distance) * 100

def process_frame(frame, data):
    frame = imutils.resize(frame, width=500)
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []
    confidence_rates = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {data["names"][i]: counts.get(data["names"][i], 0) + 1 for i in matchedIdxs}
            name = max(counts, key=counts.get)
            if current_name != name:
                current_name = name
                print(current_name)
        confidence_rate = calculate_confidence_rate(encoding, data) if name != "Unknown" else 0
        confidence_rates.append(confidence_rate)
        names.append(name)

    return boxes, names, confidence_rates

def draw_faces(frame, boxes, names, confidence_rates):
    for ((top, right, bottom, left), name, confidence) in zip(boxes, names, confidence_rates):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        text = f"{name} ({confidence:.2f}%)"
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    cv2.imshow("Facial Recognition is Running", frame)

def facial_recognition(vs, fps, data):
    start_time = time.time()
    while running:
        frame = vs.read()
        boxes, names, confidence_rates = process_frame(frame, data)
        draw_faces(frame, boxes, names, confidence_rates)
        if time.time() - start_time >= CONFIDENCE_INTERVAL:
            if confidence_rates:
                average_confidence = sum(confidence_rates) / len(confidence_rates)
                print(f"Average Confidence Rate: {average_confidence:.2f}%")
            else:
                print("No recognized faces")
            start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()

def start_facial_recognition():
    data = load_data()
    vs = start_video_stream()
    fps = start_fps_counter()
    Thread(target=facial_recognition, args=(vs, fps, data)).start()

def stop_facial_recognition():
    global running
    running = False

def on_closing(root):
    stop_facial_recognition()
    root.destroy()

def create_gui():
    root = tk.Tk()
    root.title("Facial Recognition")
    tk.Button(root, text="Start", command=start_facial_recognition).pack()
    tk.Button(root, text="Stop", command=stop_facial_recognition).pack()
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()

if __name__ == "__main__":
    create_gui()