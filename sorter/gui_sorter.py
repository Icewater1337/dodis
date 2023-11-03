import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import PyPDF2
import os
import shutil

from pdf2image import convert_from_path


class PDFSorter(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("PDF Sorter")

        self.source_folder = ""
        self.handwritten_folder = ""
        self.pdf_files = []
        self.current_pdf = None
        self.previous_pdfs = []  # Stack to keep track of viewed PDFs

        self.canvas = tk.Canvas(self, width=400, height=600)
        self.canvas.pack()

        self.button_computer = tk.Button(self, text="Computer", command=self.next_pdf)
        self.button_computer.pack(side=tk.LEFT, padx=10)

        self.button_handwritten = tk.Button(self, text="Handwritten", command=self.move_to_handwritten)
        self.button_handwritten.pack(side=tk.LEFT, padx=10)

        self.button_back = tk.Button(self, text="Back", command=self.show_previous_pdf)
        self.button_back.pack(side=tk.RIGHT, padx=10)

        self.choose_folders()

    def display_pdf(self, pdf_path):
        # Convert the first page of the PDF to an image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        image = images[0]

        # Resize the image to fit the canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_reqwidth()
        canvas_height = self.canvas.winfo_reqheight()

        img_width, img_height = image.size

        # Calculate aspect ratio
        aspect = img_width / float(img_height)

        # Based on aspect ratio decide the final width and height
        if aspect > 1:  # Landscape orientation, width is greater than height
            new_width = canvas_width
            new_height = int(new_width / aspect)
        else:  # Portrait orientation, height is greater than width
            new_height = canvas_height
            new_width = int(new_height * aspect)

        # Resize the image
        image = image.resize((new_width, new_height))

        photo = ImageTk.PhotoImage(image)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo
    def choose_folders(self):
        self.source_folder = filedialog.askdirectory(title="Choose the source folder")
        self.handwritten_folder = filedialog.askdirectory(title="Choose the handwritten folder")

        if not self.source_folder or not self.handwritten_folder:
            messagebox.showerror("Error", "Both folders must be selected")
            self.destroy()

        self.pdf_files = [f for f in os.listdir(self.source_folder) if f.endswith('.pdf')]
        if not self.pdf_files:
            messagebox.showinfo("Info", "No PDF files found in source folder")
            self.destroy()

        self.show_next_pdf()

    def show_next_pdf(self):
        if self.current_pdf:  # If there's a current PDF, push it to the previous_pdfs stack
            self.previous_pdfs.append(self.current_pdf)

        if not self.pdf_files:
            messagebox.showinfo("Info", "All PDFs processed")
            self.destroy()
            return

        self.current_pdf = os.path.join(self.source_folder, self.pdf_files.pop(0))
        self.display_pdf(self.current_pdf)

    def show_previous_pdf(self):
        if not self.previous_pdfs:
            messagebox.showinfo("Info", "No previous PDFs")
            return

        # Pop the last viewed PDF from the stack
        last_viewed_pdf = self.previous_pdfs.pop()
        if "computer" in last_viewed_pdf:
            fname = os.path.basename(last_viewed_pdf)
            shutil.move(f"{self.handwritten_folder}/{fname}", self.source_folder)
        # Add the current_pdf back to the list (since we're going back)
        if self.current_pdf:
            self.pdf_files.insert(0, os.path.basename(self.current_pdf))

        self.display_pdf(last_viewed_pdf)
        self.current_pdf = last_viewed_pdf
    def move_to_handwritten(self):
        if self.current_pdf:
            shutil.move(self.current_pdf, self.handwritten_folder)
        self.show_next_pdf()

    def next_pdf(self):
        self.show_next_pdf()


if __name__ == "__main__":
    app = PDFSorter()
    app.mainloop()
