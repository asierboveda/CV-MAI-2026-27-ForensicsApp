"""Find the image with the most matches in a folder."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .feature_matching import match_features_between


class FolderMatchingTool(ForensicsTool):
    tool_id = "folder_matching"
    title = "Find closest image in folder"
    category = "Set 5"
    description = "Find the image with the most SIFT matches after resizing."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        folder = filedialog.askdirectory(
            title="Select folder to search",
            parent=parent,
        )
        if not folder:
            return None

        extensions = {
            ".png", ".jpg", ".jpeg", ".bmp",
            ".tif", ".tiff", ".webp",
        }

        best_path = None
        best_visual = None
        best_count = 0
        compared = 0
        skipped = 0

        reference_name = (
            document.path.name if document.path else "Reference"
        )

        for path in sorted(Path(folder).iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            # Do not compare the reference file with itself.
            if document.path and path.resolve() == document.path.resolve():
                continue

            try:
                with Image.open(path) as image:
                    candidate = image.convert("RGB")

                _, _, matches, visual = match_features_between(
                    document.current,
                    candidate,
                    method="SIFT",
                    cross_check=True,
                    max_ratio=0.8,
                    title1=reference_name,
                    title2=path.name,
                    ransac_threshold=None,
                )
            except (UnidentifiedImageError, OSError, ValueError):
                skipped += 1
                continue

            compared += 1
            count = len(matches)

            if count > best_count:
                best_count = count
                best_path = path
                best_visual = visual

        if compared == 0:
            messagebox.showinfo(
                "Folder matching",
                "No images could be compared in this folder.",
                parent=parent,
            )
            return None

        if best_path is None:
            messagebox.showinfo(
                "Folder matching",
                "No matches were found with the reference image.",
                parent=parent,
            )
            return None

        return ToolResult(
            image=best_visual,
            message=(
                f"Best image: {best_path.name} "
                f"with {best_count} matches."
            ),
            details={
                "Operation": "Folder matching",
                "Method": "SIFT",
                "Max image dimension (px)": 1000,
                "Best image": best_path.name,
                "Matches found": best_count,
                "Images compared": compared,
                "Skipped images": skipped,
                "Folder": folder,
            },
        )
