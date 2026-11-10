"""Set 7 — Interactive Bag-of-Visual-Words Codebook Construction & Training.

Allows users to:
1. Select an image dataset directory structured by class subfolders.
2. Configure K (number of clusters) with dynamic range validation (1 < K < total images).
3. Extract SIFT descriptors, fit KMeans codebook, and train chi2-kernel SVM classifier.
4. Save the bundle to a user-specified folder under forensics_app/models/.
5. Visualise the generated or pre-constructed visual words codebook grid.
6. Load existing preconstructed codebooks and set them as active for the session.
"""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_utils import (
    DATASET_DIR,
    MODELS_DIR,
    build_codebook_bundle,
    find_dataset_classes,
    get_active_model_dir,
    load_visual_words_grid,
    set_active_model_dir,
    validate_model_dir,
)
from .bow_visual_words import VisualWordsWindow


class CodebookBuilderWindow(tk.Toplevel):
    """Interactive window for codebook construction, visualization, and management."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("Bag-of-Visual-Words Codebook Construction — Set 7")
        self.geometry("820x680")
        self.minsize(700, 560)

        # State variables
        default_ds = DATASET_DIR if DATASET_DIR.is_dir() else Path("")
        self.ds_path_var = tk.StringVar(value=str(default_ds) if default_ds else "")
        self.k_var = tk.StringVar(value="30")
        self.split_var = tk.StringVar(value="80%")
        self.folder_name_var = tk.StringVar(value="codebook_custom_k30")

        self.detected_classes: dict[str, list[Path]] = {}
        self.total_images = 0
        self._is_building = False

        self._build_ui()
        if default_ds and default_ds.is_dir():
            self._scan_dataset(str(default_ds))

    def _build_ui(self) -> None:
        # Header banner
        header = ttk.Frame(self, padding=(14, 12))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Construct BoW Codebook & Train Classifier",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "Build a visual-word codebook from any dataset with class subfolders.\n"
                "Extracts SIFT features, clusters into K visual words, extracts representative patches,\n"
                "and trains an SVM with precomputed Chi-squared kernel."
            ),
            font=("TkDefaultFont", 9),
            foreground="#555555",
        ).pack(anchor="w", pady=(3, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Scrollable container
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        # --- Section 1: Dataset Directory ---
        sec1 = ttk.LabelFrame(container, text="1. Dataset Directory (Folder of Class Folders)", padding=10)
        sec1.pack(fill="x", pady=(0, 10))

        ds_row = ttk.Frame(sec1)
        ds_row.pack(fill="x", pady=2)

        ttk.Entry(ds_row, textvariable=self.ds_path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(ds_row, text="Browse Dataset...", command=self._on_browse_dataset).pack(side="left")

        self.lbl_dataset_info = ttk.Label(
            sec1,
            text="Select a root directory containing class subfolders (e.g. dataset/butterfly/*.jpg)",
            font=("TkDefaultFont", 9),
            foreground="#2563eb",
        )
        self.lbl_dataset_info.pack(anchor="w", pady=(4, 0))

        # --- Section 2: Codebook Parameters ---
        sec2 = ttk.LabelFrame(container, text="2. Configuration & Cluster Count", padding=10)
        sec2.pack(fill="x", pady=(0, 10))

        grid = ttk.Frame(sec2)
        grid.pack(fill="x")

        # Row 1: K clusters
        ttk.Label(grid, text="Number of visual words (K):", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        k_box = ttk.Frame(grid)
        k_box.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        self.entry_k = ttk.Entry(k_box, textvariable=self.k_var, width=8)
        self.entry_k.pack(side="left", padx=(0, 8))
        self.entry_k.bind("<KeyRelease>", lambda _e: self._on_k_changed())

        self.lbl_k_range = ttk.Label(
            k_box,
            text="(Default: 10 × classes | Valid: 1 < K < total images)",
            font=("TkDefaultFont", 8),
            foreground="#555555",
        )
        self.lbl_k_range.pack(side="left")

        # Row 2: Split ratio
        ttk.Label(grid, text="Train / Test split ratio:").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        split_cb = ttk.Combobox(
            grid,
            textvariable=self.split_var,
            values=["70%", "80%", "85%", "90%", "100% (No test split)"],
            state="readonly",
            width=16,
        )
        split_cb.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        # Row 3: Target bundle folder name
        ttk.Label(grid, text="Save codebook folder name:").grid(
            row=2, column=0, sticky="w", padx=6, pady=4
        )
        self.entry_folder = ttk.Entry(grid, textvariable=self.folder_name_var, width=28)
        self.entry_folder.grid(row=2, column=1, sticky="w", padx=6, pady=4)

        self.lbl_save_dest = ttk.Label(
            grid,
            text=f"Location: forensics_app/models/{self.folder_name_var.get()}/",
            font=("Consolas", 8),
            foreground="#6b7280",
        )
        self.lbl_save_dest.grid(row=3, column=1, sticky="w", padx=6, pady=(0, 4))
        self.folder_name_var.trace_add(
            "write",
            lambda *_: self.lbl_save_dest.configure(
                text=f"Location: forensics_app/models/{self.folder_name_var.get()}/"
            ),
        )

        # --- Section 3: Actions & Progress ---
        sec3 = ttk.LabelFrame(container, text="3. Construction & Visualization", padding=10)
        sec3.pack(fill="x", pady=(0, 10))

        btn_bar = ttk.Frame(sec3)
        btn_bar.pack(fill="x", pady=(0, 8))

        self.btn_construct = ttk.Button(
            btn_bar,
            text="🚀 Construct & Save Codebook",
            command=self._on_construct,
        )
        self.btn_construct.pack(side="left", padx=(0, 8))

        self.btn_visualise = ttk.Button(
            btn_bar,
            text="🔍 Visualise Codebook Grid",
            command=self._on_visualise,
        )
        self.btn_visualise.pack(side="left", padx=(0, 8))

        self.btn_load_existing = ttk.Button(
            btn_bar,
            text="📂 Load Preconstructed Codebook...",
            command=self._on_load_preconstructed,
        )
        self.btn_load_existing.pack(side="left")

        # Progress bar
        self.prog_bar = ttk.Progressbar(sec3, orient="horizontal", mode="determinate")
        self.prog_bar.pack(fill="x", pady=(4, 4))

        self.lbl_status = ttk.Label(
            sec3,
            text=f"Active codebook: {get_active_model_dir().name}",
            font=("TkDefaultFont", 9),
            foreground="#16a34a",
        )
        self.lbl_status.pack(anchor="w")

        # --- Section 4: Log Output ---
        sec4 = ttk.LabelFrame(container, text="Build Output & Summary", padding=8)
        sec4.pack(fill="both", expand=True)

        self.txt_log = tk.Text(sec4, height=7, font=("Consolas", 9), background="#f8fafc", wrap="word")
        scr = ttk.Scrollbar(sec4, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        self.txt_log.pack(side="left", fill="both", expand=True)

        # Footer
        footer = ttk.Frame(self, padding=(12, 8))
        footer.pack(fill="x", side="bottom")
        ttk.Button(footer, text="Close", command=self.destroy).pack(side="right")

    def _log(self, msg: str) -> None:
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _on_browse_dataset(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Select Dataset Root Directory",
            initialdir=str(DATASET_DIR.parent if DATASET_DIR.exists() else MODELS_DIR.parent),
        )
        if folder:
            self.ds_path_var.set(folder)
            self._scan_dataset(folder)

    def _scan_dataset(self, folder_str: str) -> None:
        p = Path(folder_str)
        self.detected_classes = find_dataset_classes(p)
        self.total_images = sum(len(imgs) for imgs in self.detected_classes.values())

        if not self.detected_classes:
            self.lbl_dataset_info.configure(
                text="⚠ No class subfolders with images found in this folder.",
                foreground="#dc2626",
            )
            return

        cls_summary = ", ".join(f"{cls} ({len(imgs)})" for cls, imgs in self.detected_classes.items())
        self.lbl_dataset_info.configure(
            text=f"Found {len(self.detected_classes)} classes: {cls_summary} — Total: {self.total_images} images",
            foreground="#16a34a",
        )

        # Set default K = 10 * num_classes, but ensure valid range
        default_k = max(2, min(len(self.detected_classes) * 10, max(2, self.total_images - 1)))
        self.k_var.set(str(default_k))
        self.lbl_k_range.configure(
            text=f"(Default: 10 × {len(self.detected_classes)} classes = {default_k} | Valid range: 2 to {max(2, self.total_images - 1)})"
        )
        self.folder_name_var.set(f"codebook_{p.name}_k{default_k}")

    def _on_k_changed(self) -> None:
        k_str = self.k_var.get().strip()
        try:
            k_val = int(k_str)
            if self.total_images > 0:
                if 1 < k_val < self.total_images:
                    self.lbl_k_range.configure(
                        text=f"✓ Valid: 1 < K={k_val} < {self.total_images} total images",
                        foreground="#16a34a",
                    )
                else:
                    self.lbl_k_range.configure(
                        text=f"✗ Out of range! K must be between 2 and {self.total_images - 1}",
                        foreground="#dc2626",
                    )
        except ValueError:
            self.lbl_k_range.configure(text="✗ Must be an integer", foreground="#dc2626")

    def _on_construct(self) -> None:
        if self._is_building:
            return

        ds_path = self.ds_path_var.get().strip()
        if not ds_path or not Path(ds_path).is_dir():
            messagebox.showerror("Invalid Dataset", "Please select a valid dataset folder.", parent=self)
            return

        if not self.detected_classes or self.total_images == 0:
            self._scan_dataset(ds_path)
            if self.total_images == 0:
                messagebox.showerror("No Images", "The dataset contains no image files in class subfolders.", parent=self)
                return

        # Validate K: must be 1 < K < total_images
        try:
            k_val = int(self.k_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid K", "Cluster count K must be an integer.", parent=self)
            return

        if k_val <= 1 or k_val >= self.total_images:
            messagebox.showerror(
                "Invalid Cluster Count K",
                f"Cluster count K={k_val} must be strictly greater than 1 and less than total images ({self.total_images}).\n\n"
                f"Allowed range: 2 to {self.total_images - 1}.",
                parent=self,
            )
            return

        # Determine output folder
        folder_name = self.folder_name_var.get().strip() or f"codebook_k{k_val}"
        output_dir = MODELS_DIR / folder_name

        split_pct = int(self.split_var.get().replace("%", "").split()[0])
        split_ratio = split_pct / 100.0

        self._is_building = True
        self.btn_construct.configure(state="disabled")
        self.prog_bar["value"] = 0
        self.txt_log.delete("1.0", "end")
        self._log(f"Starting codebook construction for K={k_val}...")
        self._log(f"Dataset: {ds_path} ({self.total_images} images across {len(self.detected_classes)} classes)")
        self._log(f"Target folder: {output_dir}")

        def _worker():
            try:
                def _progress_cb(pct: float, msg: str):
                    self.after(0, lambda: self._update_progress(pct, msg))

                meta = build_codebook_bundle(
                    dataset_dir=ds_path,
                    output_dir=output_dir,
                    k=k_val,
                    split_ratio=split_ratio,
                    progress_callback=_progress_cb,
                )
                self.after(0, lambda: self._on_build_success(output_dir, meta))
            except Exception as exc:
                self.after(0, lambda: self._on_build_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_progress(self, pct: float, msg: str) -> None:
        self.prog_bar["value"] = int(pct * 100)
        self.lbl_status.configure(text=msg, foreground="#2563eb")
        self._log(f"[{int(pct*100)}%] {msg}")

    def _on_build_success(self, output_dir: Path, meta: dict[str, Any]) -> None:
        self._is_building = False
        self.btn_construct.configure(state="normal")
        self.prog_bar["value"] = 100
        self.lbl_status.configure(
            text=f"✓ Codebook '{output_dir.name}' built & set as active!",
            foreground="#16a34a",
        )
        self._log(f"\nSUCCESS: Codebook bundle saved to {output_dir}")
        self._log(f"Visual Words: {meta['k']}, Train images: {meta['num_train_images']}, SIFT descriptors: {meta['total_descriptors']:,}")
        self._log(f"Classes: {', '.join(meta['classes'])}")

        messagebox.showinfo(
            "Codebook Constructed",
            f"Successfully built codebook '{output_dir.name}' with K={meta['k']} visual words!\n\n"
            f"It is now set as the active model for BoW Histogram, Classification, and Retrieval.",
            parent=self,
        )

    def _on_build_error(self, err_msg: str) -> None:
        self._is_building = False
        self.btn_construct.configure(state="normal")
        self.lbl_status.configure(text="✗ Build failed", foreground="#dc2626")
        self._log(f"\nERROR: {err_msg}")
        messagebox.showerror("Build Error", f"Failed to build codebook:\n\n{err_msg}", parent=self)

    def _on_visualise(self) -> None:
        active_dir = get_active_model_dir()
        try:
            grid_img = load_visual_words_grid(active_dir)
            VisualWordsWindow(self, grid_img, num_words=int(self.k_var.get() or "30"))
        except Exception as exc:
            messagebox.showerror(
                "Visualisation Error",
                f"Could not load visual words grid from {active_dir.name}:\n\n{exc}",
                parent=self,
            )

    def _on_load_preconstructed(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Select Preconstructed Codebook Folder",
            initialdir=str(MODELS_DIR),
        )
        if not folder:
            return

        valid, msg = validate_model_dir(folder)
        if not valid:
            messagebox.showerror("Invalid Codebook", msg, parent=self)
            return

        target = Path(folder).resolve()
        set_active_model_dir(target)
        self.lbl_status.configure(
            text=f"Active codebook set to: {target.name}",
            foreground="#16a34a",
        )
        self._log(f"\nLoaded preconstructed codebook: {target}")
        messagebox.showinfo(
            "Codebook Loaded",
            f"Loaded codebook bundle from '{target.name}'.\nIt is now the active codebook for Set 7 tools.",
            parent=self,
        )


class CodebookBuilderTool(ForensicsTool):
    """Set 7 — Construct BoW Codebook and Train Classifier."""

    tool_id = "bow_build"
    title = "Construct Codebook (BoW)"
    category = "Set 7"
    description = (
        "Construct a visual-word codebook from class folders, visualize the codebook, "
        "and train the SVM classifier."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        CodebookBuilderWindow(parent)
        active_dir = get_active_model_dir()
        return ToolResult(
            image=None,
            message=f"Opened BoW Codebook Construction dialog. Active codebook: {active_dir.name}",
            details={
                "Operation": "BoW Codebook Construction",
                "Active Codebook": active_dir.name,
                "Model Directory": str(active_dir),
            },
        )

