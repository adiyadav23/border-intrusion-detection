import customtkinter as ctk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import os
import time
from datetime import datetime
import numpy as np
from tkinter import filedialog, ttk, messagebox
import sqlite3
import hashlib

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.5
BORDER_Y = 200
SAVE_DIR = "evidence"

FRAME_SKIP = 4
UI_UPDATE_SKIP = 3

os.makedirs(SAVE_DIR, exist_ok=True)

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# ================= DATABASE =================
conn = sqlite3.connect("detections.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    image_path TEXT,
    event_type TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()

# ================= MODEL =================
model = YOLO(MODEL_PATH)

try:
    model.to("cuda")   # 🔥 GPU
except:
    model.to("cpu")

# ================= AUTH =================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ================= LOGIN =================
class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Login - Intrusion Detection System")
        self.geometry("400x350")

        ctk.CTkLabel(self, text="🔐 Login",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        self.username = ctk.CTkEntry(self, placeholder_text="Username")
        self.username.pack(pady=10)

        self.password = ctk.CTkEntry(self, placeholder_text="Password", show="*")
        self.password.pack(pady=10)

        ctk.CTkButton(self, text="Login", command=self.login).pack(pady=10)
        ctk.CTkButton(self, text="Register", command=self.register).pack(pady=5)

    def login(self):
        user = self.username.get()
        pwd = hash_password(self.password.get())

        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        if cursor.fetchone():
            self.destroy()
            App().mainloop()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def register(self):
        user = self.username.get()
        pwd = hash_password(self.password.get())

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, pwd))
            conn.commit()
            messagebox.showinfo("Success", "User registered!")
        except:
            messagebox.showerror("Error", "Username already exists")

# ================= MAIN APP =================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Intrusion Detection System")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")

        self.video_path = None
        self.cap = None
        self.running = False
        self.last_saved = 0
        self.history_visible = False

        # ===== SIDEBAR =====
        self.sidebar = ctk.CTkFrame(self, width=250)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="🚨 AI Security",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        def styled_button(text, command, color="#2563eb"):
            return ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=45,
                corner_radius=12,
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color=color,
                hover_color="#1d4ed8"
            )

        styled_button("📁 Upload Video", self.upload_video).pack(pady=10, padx=20, fill="x")
        self.start_btn = styled_button("🚀 Start Detection", self.start)
        self.start_btn.pack(pady=10, padx=20, fill="x")
        styled_button("⏹ Stop", self.stop, "#dc2626").pack(pady=10, padx=20, fill="x")
        styled_button("📂 History", self.toggle_history).pack(pady=10, padx=20, fill="x")

        self.status = ctk.CTkLabel(self.sidebar, text="Status: Idle", text_color="gray")
        self.status.pack(pady=20)

        # ===== MAIN =====
        self.main = ctk.CTkFrame(self)
        self.main.pack(side="right", expand=True, fill="both")

        ctk.CTkLabel(self.main, text="Intrusion Detection System",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        self.video_label = ctk.CTkLabel(self.main, text="")
        self.video_label.pack(pady=10)

        # ===== HISTORY =====
        self.history_frame = ctk.CTkFrame(self.main)

        self.tree = ttk.Treeview(
            self.history_frame,
            columns=("ID", "Time", "Event"),
            show="headings",
            height=6
        )
        self.tree.heading("ID", text="ID")
        self.tree.heading("Time", text="Timestamp")
        self.tree.heading("Event", text="Event")
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<Double-1>", self.open_image)

    # ================= FUNCTIONS =================
    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.status.configure(text=f"Loaded: {os.path.basename(path)}", text_color="cyan")

    def start(self):
        if not self.video_path:
            self.status.configure(text="⚠ Upload video first", text_color="orange")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        self.status.configure(text="Running...", text_color="green")

        threading.Thread(target=self.process, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

        self.status.configure(text="Stopped", text_color="red")

    def is_green(self, frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        return (np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])) > 0.3

    def process(self):
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % (FRAME_SKIP + 1) != 0:
                continue

            frame = cv2.resize(frame, (512, 288))

            results = model(frame, classes=[2, 5, 7], stream=True, half=True)

            intrusion = False

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if y2 <= BORDER_Y:
                        continue

                    if self.is_green(frame, x1, y1, x2, y2):
                        continue

                    intrusion = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "INTRUSION!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if intrusion:
                now = time.time()
                if now - self.last_saved > 2:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    filename = f"{SAVE_DIR}/intrusion_{datetime.now().strftime('%H%M%S')}.jpg"

                    cv2.imwrite(filename, frame)

                    cursor.execute(
                        "INSERT INTO detections (timestamp, image_path, event_type) VALUES (?, ?, ?)",
                        (timestamp, filename, "Intrusion")
                    )
                    conn.commit()

                    self.status.configure(text="🚨 Intrusion Detected!", text_color="yellow")
                    self.last_saved = now

            if frame_count % UI_UPDATE_SKIP == 0:
                self.update_ui(frame)

        self.stop()

    def update_ui(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    def toggle_history(self):
        if self.history_visible:
            self.history_frame.pack_forget()
            self.history_visible = False
        else:
            self.history_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.load_data()
            self.history_visible = True

    def load_data(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        cursor.execute("SELECT id, timestamp, event_type FROM detections ORDER BY id DESC")
        for row in cursor.fetchall():
            self.tree.insert("", "end", values=row)

    def open_image(self, event):
        item = self.tree.item(self.tree.focus())
        if not item:
            return

        rid = item["values"][0]
        cursor.execute("SELECT image_path FROM detections WHERE id=?", (rid,))
        res = cursor.fetchone()

        if res:
            os.startfile(res[0])

# ================= RUN =================
if __name__ == "__main__":
    LoginWindow().mainloop()
