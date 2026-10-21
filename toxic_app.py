import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import tkinter as tk
from tkinter import messagebox
import joblib
import os

MODEL_PATH = "toxicity_model.pkl"


# =============== تحميل الموديل بشكل آمن ===============
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Error",
            f"File '{MODEL_PATH}' not found.\n"
            "Put the model file in the SAME folder as toxic_app.py."
        )
        return None

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        messagebox.showerror(
            "Model Load Error",
            f"Cannot load model:\n\n{e}"
        )
        return None


# =============== بناء الواجهة ===============
root = tk.Tk()
root.title("Toxic Comment Classifier")
root.geometry("550x380")
root.configure(bg="white")   # الخلفية الأساسية

title_label = tk.Label(
    root,
    text="Toxic Comment Classifier",
    font=("Arial", 16, "bold"),
    bg="white"
)
title_label.pack(pady=10)

input_label = tk.Label(root, text="Enter a comment:", bg="white")
input_label.pack()

text_box = tk.Text(root, height=6, width=60)
text_box.pack(pady=5)

result_label = tk.Label(root, text="Result: ", font=("Arial", 14), bg="white")
result_label.pack(pady=5)

prob_label = tk.Label(root, text="Toxicity: -", font=("Arial", 12), bg="white")
prob_label.pack(pady=2)

emoji_label = tk.Label(root, text="", font=("Segoe UI Emoji", 32), bg="white")
emoji_label.pack(pady=5)

model = None  # نحمّله أول مرة عند الضغط على الزر


# =============== دالة تغيير الخلفية لكل العناصر ===============
def set_background(color: str):
    root.config(bg=color)
    title_label.config(bg=color)
    input_label.config(bg=color)
    result_label.config(bg=color)
    prob_label.config(bg=color)
    emoji_label.config(bg=color)
    # نخلي مربع النص أبيض عشان القراءة
    text_box.config(bg="white")


# =============== دالة التنبؤ ===============
def predict_toxicity():
    global model

    if model is None:
        model = load_model_safely()
        if model is None:
            return

    text = text_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Warning", "Please enter a comment.")
        return

    try:
        # التنبؤ: 0 أو 1
        pred = model.predict([text])[0]

        # نسب الاحتمال
        proba = model.predict_proba([text])[0]

        # نجيب احتمال الكلاس 1 (toxic) بشكل آمن حتى لو ترتيب الكلاسات مو [0,1]
        import numpy as np
        classes = list(model.classes_)
        idx_toxic = classes.index(1)  # نفترض إن 1 = toxic
        prob_toxic = float(proba[idx_toxic])
        toxicity_percent = prob_toxic * 100.0

    except Exception as e:
        messagebox.showerror("Prediction Error", f"{e}")
        return

    # نحدّث النتيجة + اللون + الإيموجي
    if pred == 1:
        # توكسيك
        result_label.config(text="Result: Toxic", fg="red")
        prob_label.config(text=f"Toxicity: {toxicity_percent:.1f}%")
        emoji_label.config(text="😡")  # إيموجي زعلان
        set_background("light coral")
    else:
        # نون توكسيك
        result_label.config(text="Result: Non-toxic", fg="green")
        prob_label.config(text=f"Toxicity: {toxicity_percent:.1f}%")
        emoji_label.config(text="😄")  # إيموجي مبسوط
        set_background("pale green")


# =============== دالة زر Clear ===============
def clear_all():
    text_box.delete("1.0", tk.END)
    result_label.config(text="Result: ", fg="black")
    prob_label.config(text="Toxicity: -")
    emoji_label.config(text="")
    set_background("white")


# الأزرار
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_toxicity)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_all)
clear_button.grid(row=0, column=1, padx=10)

root.mainloop()
