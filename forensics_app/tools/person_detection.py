"""Person detection using HOG."""

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person detection using HOG"
    category = "Set 4"
    description = "Locate a person by comparing HOG descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument,) -> ToolResult | None:
        assert document.current is not None

        image = document.current.convert("RGB")
        filename = filedialog.askopenfilename(
            title = "Select the person template",
            parent = parent,
        )
        if not filename:
            return None
        with Image.open(filename) as source:
            template = source.convert("RGB")
        image_array = np.asarray(image)
        template_array = np.asarray(template)

        height, width = template_array.shape[:2]
        image_height, image_width = image_array.shape[:2]
        
        hog_parameters = {
            "orientations": 10,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "channel_axis": -1,
        }
        template_features = hog(template_array, **hog_parameters)

        best_distance = float("inf")
        best_product = -float("inf")
        distance_position = (0,0)
        product_position = (0,0)
        step_size = 8
        for y in range(0, image_height - height + 1, step_size):
            for x in range(0, image_width - width + 1, step_size):
                region = image_array[y:y + height, x:x + width]
                features = hog(region, **hog_parameters)

                distance = np.linalg.norm(template_features - features)
                product = np.dot(template_features, features)

                if distance < best_distance:
                    best_distance = distance
                    distance_position = (x, y)

                if product > best_product:
                    best_product = product
                    product_position = (x, y)

        output = image.copy()
        draw = ImageDraw.Draw(output)

        x, y = distance_position
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline="red",
            width=2,
        )

        x, y = product_position
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline="cyan",
            width=2,
        )

        return ToolResult(
            image=output,
            message="Red: Euclidean distance, cyan: dot product."
        )
