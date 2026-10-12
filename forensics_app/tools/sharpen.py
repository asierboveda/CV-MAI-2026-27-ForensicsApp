import math
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Image sharpening"
    category = "Set 3"
    description = "Sharpen the working image with a configurable amount."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        while True:
            amount = simpledialog.askfloat(
                self.title, "Sharpening amount:", parent=parent, initialvalue=1.0, minvalue=0.0
            )
            if amount is None:
                return None
            if math.isfinite(amount) and amount >= 0:
                break
            messagebox.showerror("Invalid amount", "Amount must be finite and non-negative.", parent=parent)

        is_color = document.current.mode not in {"1", "L", "I", "F"}
        pixels = np.array(document.current.convert("RGB" if is_color else "L"), dtype=np.float64)
        # Alternativa manual estudiada: desenfocar, extraer el detalle y sumarlo.
        # Usamos el mismo radio y tratamiento de canales que en unsharp_mask.
        # mode="reflect" iguala también el tratamiento de los bordes de la imagen.
        # blurred = filters.gaussian(
        #     pixels, sigma=1.0, mode="reflect", preserve_range=True,
        #     channel_axis=2 if is_color else None,
        # )
        # detail = pixels - blurred
        # sharpened = pixels + amount * detail

        # Versión utilizada: unsharp_mask realiza ese mismo cálculo internamente.
        # El ejemplo del PDF usa radius=2 y amount=1.5; aquí el radio es 1.0
        # y el usuario configura amount. Ambas alternativas comparten el clip final.
        sharpened = filters.unsharp_mask(
            pixels, radius=1.0, amount=amount, preserve_range=True,
            channel_axis=2 if is_color else None,
        )
        output = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
        return ToolResult(
            image=output,
            message=f"Sharpened image (amount {amount:g}).",
            details={"Operation": "Sharpen", "Amount": amount, "Radius": 1.0},
        )
