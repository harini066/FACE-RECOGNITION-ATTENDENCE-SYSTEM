import tkinter as tk
from tkinter import ttk, messagebox as mess, simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
from cryptography.fernet import Fernet

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    mess.showinfo(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com'")

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror(title='File missing', message='Please contact us for help')
        window.destroy()

def generate_key():
    return Fernet.generate_key()

def encrypt_password(password, key):
    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(password.encode())

def decrypt_password(encrypted_password, key):
    cipher_suite = Fernet(key)
    return cipher_suite.decrypt(encrypted_password).decode()

def save_key(key):
    with open("key.key", "wb") as key_file:
        key_file.write(key)

def load_key():
    return open("key.key", "rb").read()

def save_password():
    assure_path_exists("TrainingImageLabel/")
    key = load_key() if os.path.exists("key.key") else generate_key()
    encrypted_password = encrypt_password(new_password.get(), key)
    with open("TrainingImageLabel/psd.txt", "wb") as password_file:
        password_file.write(encrypted_password)
    save_key(key)
    mess.showinfo(title='Password', message='Password registered successfully!')

def change_password():
    global new_password, old_password, confirm_password
    master = tk.Tk()
    master.geometry("400x200")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")

    tk.Label(master, text='Enter Old Password', bg='white', font=('times', 12, 'bold')).place(x=10, y=10)
    old_password = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, 'bold'), show='*')
    old_password.place(x=180, y=10)

    tk.Label(master, text='Enter New Password', bg='white', font=('times', 12, 'bold')).place(x=10, y=45)
    new_password = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, 'bold'), show='*')
    new_password.place(x=180, y=45)

    tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, 'bold')).place(x=10, y=80)
    confirm_password = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, 'bold'), show='*')
    confirm_password.place(x=180, y=80)

    tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25, font=('times', 10, 'bold')).place(x=200, y=120)
    tk.Button(master, text="Save", command=save_password, fg="black", bg="#3ece48", height=1, width=25, font=('times', 10, 'bold')).place(x=10, y=120)

    master.mainloop()

def psw():
    if not os.path.exists("key.key"):
        key = generate_key()
        save_key(key)
        new_password = tsd.askstring('Password Setup', 'Please enter a new password', show='*')
        if new_password:
            encrypted_password = encrypt_password(new_password, key)
            with open("TrainingImageLabel/psd.txt", "wb") as password_file:
                password_file.write(encrypted_password)
            mess.showinfo(title='Password Registered', message='Password registered successfully!')
        else:
            mess.showerror(title='Error', message='Password not set')
    else:
        key = load_key()
        with open("TrainingImageLabel/psd.txt", "rb") as password_file:
            encrypted_password = password_file.read()
        stored_password = decrypt_password(encrypted_password, key)
        entered_password = tsd.askstring('Password', 'Enter Password', show='*')
        if entered_password == stored_password:
            TrainImages()
        else:
            mess.showerror(title='Error', message='Wrong password')

def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')
    message1.config(text="1)Take Images  >>>  2)Save Profile")

def take_images_thread():
    try:
        Id = txt.get()
        name = txt2.get()
        if not (name.isalpha() or ' ' in name):
            mess.showerror(title='Error', message='Enter a valid name')
            return
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            mess.showerror(title='Error', message='Could not access the camera')
            return
        
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        sample_num = 0
        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sample_num += 1
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q') or sample_num >= 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        if sample_num > 0:
            row = [Id, name]
            with open('StudentDetails/StudentDetails.csv', 'a+') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)
            message1.config(text=f"Images Taken for ID: {Id}")
        else:
            mess.showerror(title='Error', message='No faces detected')
    except Exception as e:
        mess.showerror(title='Error', message=str(e))

def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    threading.Thread(target=take_images_thread).start()

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces, Ids = getImagesAndLabels("TrainingImage")
    
    if faces:
        recognizer.train(faces, np.array(Ids))
        recognizer.save("TrainingImageLabel/Trainer.yml")
        mess.showinfo(title='Success', message='Profile saved successfully!')
        message.config(text=f'Total Registrations till now: {len(set(Ids))}')
    else:
        mess.showerror(title='No Registrations', message='Please register someone first')

def getImagesAndLabels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    Ids = []
    
    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            Id = int(os.path.split(image_path)[-1].split(".")[1])
            face_samples.append(image_np)
            Ids.append(Id)
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
    return face_samples, Ids

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists("TrainingImageLabel/Trainer.yml"):
        recognizer.read("TrainingImageLabel/Trainer.yml")
    else:
        mess.showerror(title='Data Missing', message='Please click on Save Profile to reset data')
        return
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50:
                profile = getProfile(Id)
                if profile:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, str(profile[1]), (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    markAttendance(Id, profile[1])
                else:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def getProfile(Id):
    with open('StudentDetails/StudentDetails.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0] == str(Id):
                return row
    return None

def markAttendance(Id, name):
    with open('Attendance/Attendance.csv', 'a+') as csv_file:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        writer = csv.writer(csv_file)
        writer.writerow([Id, name, timestamp])

# GUI Setup
window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.geometry('1280x720')
window.configure(background='white')

# Frames
frame1 = tk.Frame(window, bg="white")
frame1.place(relx=0.1, rely=0.17, relwidth=0.8, relheight=0.80)

frame2 = tk.Frame(window, bg="white")
frame2.place(relx=0.1, rely=0.02, relwidth=0.8, relheight=0.12)

# Clock
clock = tk.Label(frame2, fg="green", bg="white", font=('times', 20, 'bold'))
clock.place(relx=0.8, rely=0)
tick()

# UI Elements
head1 = tk.Label(frame2, text="Face Recognition Based Attendance System", fg="black", bg="white", font=('times', 29, 'bold'))
head1.place(x=10, y=10)

lbl = tk.Label(frame1, text="Enter ID", width=20, height=2, fg="black", bg="white", font=('times', 15, 'bold'))
lbl.place(x=200, y=50)

txt = tk.Entry(frame1, width=32, fg="black", font=('times', 15, 'bold'), bg="white")
txt.place(x=500, y=55)

lbl2 = tk.Label(frame1, text="Enter Name", width=20, fg="black", bg="white", height=2, font=('times', 15, 'bold'))
lbl2.place(x=200, y=120)

txt2 = tk.Entry(frame1, width=32, fg="black", font=('times', 15, 'bold'), bg="white")
txt2.place(x=500, y=130)

message1 = tk.Label(frame1, text="1)Take Images  >>>  2)Save Profile", bg="white", fg="black", width=39, height=1, activebackground="white", font=('times', 15, 'bold'))
message1.place(x=7, y=230)

message = tk.Label(frame1, text="", bg="white", fg="black", width=39, height=1, activebackground="white", font=('times', 16, 'bold'))
message.place(x=7, y=450)

# Buttons
clearButton = tk.Button(frame1, text="Clear", command=clear, fg="white", bg="#13059c", width=11, height=1, activebackground="white", font=('times', 15, 'bold'))
clearButton.place(x=950, y=50)

takeImg = tk.Button(frame1, text="Take Images", command=TakeImages, fg="white", bg="#13059c", width=14, height=1, activebackground="white", font=('times', 15, 'bold'))
takeImg.place(x=200, y=300)

trainImg = tk.Button(frame1, text="Save Profile", command=psw, fg="white", bg="#13059c", width=14, height=1, activebackground="white", font=('times', 15, 'bold'))
trainImg.place(x=500, y=300)

trackImg = tk.Button(frame1, text="Track Images", command=TrackImages, fg="white", bg="#13059c", width=14, height=1, activebackground="white", font=('times', 15, 'bold'))
trackImg.place(x=800, y=300)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="white", bg="#13059c", width=14, height=1, activebackground="white", font=('times', 15, 'bold'))
quitWindow.place(x=500, y=380)

window.mainloop()
