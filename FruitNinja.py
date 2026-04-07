import os
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

DATASET_PATH = "Project_Dataset"
FRESH_PATH = os.path.join(DATASET_PATH, "FreshFruits")
ROTTEN_PATH = os.path.join(DATASET_PATH, "RottenFruits")
MODEL_PATH = "fruit_cnn_classifier.keras"
IMG_SIZE = 128

def cv_to_tk(cv_img, resize_dim=(350, 350)):
    """Converts an OpenCV BGR image to a Tkinter-compatible PhotoImage."""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pil_img.thumbnail(resize_dim, Image.Resampling.LANCZOS) 
    tk_img = ImageTk.PhotoImage(pil_img)
    return tk_img

class FruitCNNClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Deep Learning Fruit Classifier (CNN)")
        master.geometry("800x850") 

        self.train_frame = LabelFrame(master, text="CNN Training Control", padx=10, pady=10)
        self.train_frame.pack(fill="x", padx=10, pady=5)

        self.log_text = Text(self.train_frame, width=90, height=8, font=("Consolas", 9))
        self.log_text.pack(pady=5)

        self.progress = ttk.Progressbar(self.train_frame, orient=HORIZONTAL, length=700, mode='determinate')
        self.progress.pack(pady=5)

        self.train_button = Button(self.train_frame, text="Start CNN Training", command=self.start_training, bg="#e1e1e1")
        self.train_button.pack(pady=5)

        self.test_frame = LabelFrame(master, text="Batch Testing & Inference", padx=10, pady=10)
        self.test_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.control_frame = Frame(self.test_frame)
        self.control_frame.pack(fill="x", pady=10)
        
        self.test_btn_images = Button(self.control_frame, text="Upload Images", command=self.upload_images, state=DISABLED, bg="#dddddd")
        self.test_btn_images.pack(side=LEFT, padx=5)
        
        self.test_btn_folder = Button(self.control_frame, text="Upload Folder", command=self.upload_folder, state=DISABLED, bg="#dddddd")
        self.test_btn_folder.pack(side=LEFT, padx=5)
        self.btn_prev = Button(self.control_frame, text="< Prev", command=self.show_prev, state=DISABLED)
        self.btn_prev.pack(side=RIGHT, padx=5)
        self.count_label = Label(self.control_frame, text="0 / 0", font=("Helvetica", 10, "bold"))
        self.count_label.pack(side=RIGHT, padx=5)
        self.btn_next = Button(self.control_frame, text="Next >", command=self.show_next, state=DISABLED)
        self.btn_next.pack(side=RIGHT, padx=5)

        self.result_label = Label(self.test_frame, text="Model not ready.", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=(10, 10))

        self.image_container = Frame(self.test_frame)
        self.image_container.pack(fill="both", expand=True)
        self.lbl_image = Label(self.image_container, bg="black")
        self.lbl_image.pack(expand=True, fill=BOTH, padx=20, pady=10)

        self.model = None
        self.classes = ["Fresh", "Rotten"]
        self.batch_results = []
        self.current_index = 0

        self.check_existing_model()

    def check_existing_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                self.log("System Ready: Pre-trained CNN model loaded.")
                self.enable_testing_ui()
            except Exception as e:
                self.log(f"Error loading model: {e}")

    def enable_testing_ui(self):
        self.test_btn_images.config(state=NORMAL, bg="#4CAF50", fg="white")
        self.test_btn_folder.config(state=NORMAL, bg="#2196F3", fg="white") # Blue button
        self.result_label.config(text="Model Ready. Please upload images or a folder.", fg="green")

    def log(self, msg):
        self.log_text.insert(END, ">> " + msg + "\n")
        self.log_text.see(END)

    def start_training(self):
        self.train_button.config(state=DISABLED)
        self.test_button.config(state=DISABLED)
        threading.Thread(target=self.train_model, daemon=True).start()

    def build_cnn(self):
        """Creates the Convolutional Neural Network Architecture."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        try:
            X, y = [], []
            if not os.path.exists(FRESH_PATH) or not os.path.exists(ROTTEN_PATH):
                messagebox.showerror("Error", "Dataset folders not found! Please ensure 'Project_Dataset/FreshFruits' and 'Project_Dataset/RottenFruits' exist.")
                self.train_button.config(state=NORMAL)
                return

            fresh_files = [f for f in os.listdir(FRESH_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            rotten_files = [f for f in os.listdir(ROTTEN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total = len(fresh_files) + len(rotten_files)
            self.progress['maximum'] = total

            self.log(f"Loading {len(fresh_files)} Fresh, {len(rotten_files)} Rotten images into memory...")

            for i, f in enumerate(fresh_files):
                img = cv2.imread(os.path.join(FRESH_PATH, f))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(0)
                self.progress['value'] = i + 1

            for i, f in enumerate(rotten_files):
                img = cv2.imread(os.path.join(ROTTEN_PATH, f))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(1)
                self.progress['value'] = len(fresh_files) + i + 1

            self.log("Normalizing image data...")
            X = np.array(X, dtype="float32") / 255.0
            y = np.array(y)

            self.log("Splitting dataset into Training and Validation sets...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            self.log("Building and Compiling CNN Model...")
            self.model = self.build_cnn()

            self.log("Training CNN... (Check console for epoch progress)")
            history = self.model.fit(
                X_train, y_train, 
                validation_data=(X_test, y_test), 
                epochs=10, 
                batch_size=32,
                verbose=1
            )
            
            val_acc = history.history['val_accuracy'][-1]
            self.log(f"Training Complete. Validation Accuracy: {val_acc:.4f}")

            self.model.save(MODEL_PATH)
            self.log("Model saved to disk.")
            
            self.train_button.config(state=NORMAL)
            self.enable_testing_ui()
            messagebox.showinfo("Done", f"Training Complete.\nValidation Accuracy: {val_acc:.2%}")

        except Exception as e:
            self.log(f"Error during training: {e}")
            print(e)
            self.train_button.config(state=NORMAL)

    def upload_images(self):
        paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not paths: return
        self.start_batch(paths)

    def upload_folder(self):
        folder_path = filedialog.askdirectory(title="Select Kaggle Test Folder")
        if not folder_path: return

        self.log("Scanning folder for images... Please wait.")
        paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    paths.append(os.path.join(root_dir, file))

        if not paths:
            messagebox.showwarning("No Images", "No valid images were found.")
            return
            
        self.start_batch(paths)

    def start_batch(self, paths):
        self.test_btn_images.config(state=DISABLED)
        self.test_btn_folder.config(state=DISABLED)
        self.batch_results = []
        self.log(f"Found {len(paths)} images. Running CNN Inference...")
        threading.Thread(target=self.process_batch, args=(paths,), daemon=True).start()

    def process_batch(self, paths):
        self.progress['maximum'] = len(paths)
        self.progress['value'] = 0
        
        for i, path in enumerate(paths):
            try:
                img = cv2.imread(path)
                if img is None: continue

                tk_img = cv_to_tk(img, (400, 400))

                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized.astype("float32") / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)

                pred_prob = self.model.predict(img_batch, verbose=0)[0][0]
                
                pred_idx = 1 if pred_prob > 0.5 else 0
                pred_label = self.classes[pred_idx]
                
                prob = pred_prob if pred_idx == 1 else (1 - pred_prob)

                parent_folder = os.path.basename(os.path.dirname(path)).lower()
                
                if 'fresh' in parent_folder:
                    gt = "Fresh"
                elif 'rotten' in parent_folder:
                    gt = "Rotten"
                else:
                    fname = os.path.basename(path).lower()
                    gt = "Fresh" if fname.startswith('f') else ("Rotten" if fname.startswith('r') else "Unknown")

                self.batch_results.append({
                    "img_display": tk_img,
                    "filename": os.path.basename(path),
                    "pred": pred_label,
                    "prob": prob,
                    "gt": gt
                })
                
                self.progress['value'] = i + 1

            except Exception as e:
                print(f"Error on {path}: {e}")

        self.master.after(0, self.finish_processing)

    def finish_processing(self):
        if self.batch_results:
            self.current_index = 0
            self.update_display()
            self.log("Batch processing finished.")

            correct_count = 0
            total_valid = 0
            
            for res in self.batch_results:
                if res['gt'] != "Unknown":
                    total_valid += 1
                    if res['pred'] == res['gt']:
                        correct_count += 1
            
            if total_valid > 0:
                acc = (correct_count / total_valid) * 100
                msg = f"Batch Complete.\nImages: {len(self.batch_results)}\nValid GT: {total_valid}\nAccuracy: {acc:.2f}%"
                messagebox.showinfo("Batch Accuracy", msg)
                self.log(f"BATCH RESULT: Accuracy = {acc:.2f}% ({correct_count}/{total_valid})")
            else:
                self.log("BATCH RESULT: No identifiable Ground Truth in filenames (start with f/r).")

        else:
            self.log("No valid images processed.")
        
        self.test_btn_images.config(state=NORMAL)
        self.test_btn_folder.config(state=NORMAL)
        self.progress['value'] = 0

    def update_display(self):
        if not self.batch_results: return
        
        data = self.batch_results[self.current_index]
        
        self.lbl_image.config(image=data['img_display'])
        self.lbl_image.image = data['img_display']

        match_color = "green" if data['pred'] == data['gt'] else "red"
        if data['gt'] == "Unknown": match_color = "blue"

        info_text = (
            f"File: {data['filename']} | Ground Truth: {data['gt']}\n"
            f"CNN Prediction: {data['pred']} ({data['prob']*100:.2f}% Confidence)"
        )
        self.result_label.config(text=info_text, fg=match_color)
        
        self.count_label.config(text=f"{self.current_index + 1} / {len(self.batch_results)}")
        self.btn_prev.config(state=NORMAL if self.current_index > 0 else DISABLED)
        self.btn_next.config(state=NORMAL if self.current_index < len(self.batch_results)-1 else DISABLED)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def show_next(self):
        if self.current_index < len(self.batch_results) - 1:
            self.current_index += 1
            self.update_display()

if __name__ == "__main__":
    root = Tk()
    app = FruitCNNClassifierGUI(root)
    root.mainloop()
