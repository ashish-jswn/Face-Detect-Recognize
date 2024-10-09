import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import main
import cv2

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection and Recognition")
        
        self.label = tk.Label(root, text="Select an image to recognize faces:")
        self.label.pack(pady=10)
        
        self.btn_select = tk.Button(root, text="Select Image", command=self.select_image)
        self.btn_select.pack(pady=5)
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        image, results = main.recognize_faces(file_path)
        
        # Convert image to PIL format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Resize image to fit within the canvas while maintaining aspect ratio
        canvas_width = self.root.winfo_width()
        canvas_height = self.root.winfo_height()
        pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Resize canvas to fit the image
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        
        # Convert PIL image to ImageTk format
        imgtk = ImageTk.PhotoImage(image=pil_image)
        
        # Update canvas with the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk
        
        # Display results
        if results:
            result_text = "\n".join([f"{name}" for _, _, _, _, name in results])
            messagebox.showinfo("Recognition Results", result_text)
        else:
            messagebox.showinfo("Recognition Results", "No faces recognized.")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")  # Set initial window size
    app = FaceRecognitionApp(root)
    root.mainloop()
