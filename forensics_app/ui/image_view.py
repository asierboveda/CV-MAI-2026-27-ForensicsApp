"""Canvas that displays a PIL image scaled to the available space with a download button."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


class ImageView(ttk.Frame):
    def __init__(self, parent: tk.Misc, default_filename: str = "image.png") -> None:
        super().__init__(parent, padding=4)
        self.default_filename = default_filename
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None

        # Download / Save button in upper right corner
        self.download_button = ttk.Button(
            self,
            text="⬇ Save image",
            command=self._save_image_to_file,
        )
        self.download_button.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)
        self.download_button.configure(state="disabled")

        self.show(None)

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        if image is not None:
            self.download_button.configure(state="normal")
            self.download_button.lift()
        else:
            self.download_button.configure(state="disabled")
        self._render()

    def set_default_filename(self, filename: str) -> None:
        self.default_filename = filename

    def _save_image_to_file(self) -> None:
        if self._source is None:
            return
        initial = Path(self.default_filename).name if self.default_filename else "saved_image.png"
        filename = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            initialfile=initial,
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("TIFF image", "*.tiff"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if not filename:
            return
        try:
            self._source.save(filename)
            messagebox.showinfo("Image Saved", f"Successfully saved to {Path(filename).name}", parent=self)
        except Exception as error:
            messagebox.showerror("Save Error", str(error), parent=self)

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        if self._source is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 1),
                max(self.canvas.winfo_height() // 2, 1),
                text="No image displayed",
                fill="#c7ccd4",
                font=("TkDefaultFont", 15),
            )
            self._photo = None
            return

        available = (max(self.canvas.winfo_width() - 24, 1), max(self.canvas.winfo_height() - 24, 1))
        preview = self._source.copy()
        preview.thumbnail(available, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self._photo,
            anchor="center",
        )

