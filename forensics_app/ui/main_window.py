"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView
import numpy as np
from PIL import Image

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar(value="Ready. Open an image to begin.")

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef1f5")
        style.configure("Category.TLabel", background="#eef1f5", font=("TkDefaultFont", 10, "bold"))
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))

        self.body = ttk.Panedwindow(container, orient="horizontal")
        self.body.pack(fill="both", expand=True)

        sidebar_container = ttk.Frame(self.body, style="Sidebar.TFrame", width=255)
        sidebar_container.pack_propagate(False)
        self.body.add(sidebar_container, weight=0)

        title_frame = ttk.Frame(sidebar_container, style="Sidebar.TFrame", padding=(12, 10, 12, 4))
        title_frame.pack(fill="x")
        ttk.Label(title_frame, text="Forensics tools", style="Title.TLabel", background="#eef1f5").pack(anchor="w")

        sidebar_canvas = tk.Canvas(
            sidebar_container,
            background="#eef1f5",
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_content = ttk.Frame(sidebar_canvas, style="Sidebar.TFrame", padding=(12, 2, 8, 12))

        canvas_window = sidebar_canvas.create_window((0, 0), window=sidebar_content, anchor="nw")

        def _on_content_configure(_event: tk.Event) -> None:
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            sidebar_canvas.itemconfig(canvas_window, width=event.width)

        sidebar_content.bind("<Configure>", _on_content_configure)
        sidebar_canvas.bind("<Configure>", _on_canvas_configure)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        # Mousewheel scrolling when hovering over sidebar
        def _bind_mousewheel(event: tk.Event) -> None:
            sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event: tk.Event) -> None:
            sidebar_canvas.unbind_all("<MouseWheel>")

        def _on_mousewheel(event: tk.Event) -> None:
            sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        sidebar_container.bind("<Enter>", _bind_mousewheel)
        sidebar_container.bind("<Leave>", _unbind_mousewheel)

        scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)

        for category, tools in self.registry.categories():
            ttk.Label(sidebar_content, text=category, style="Category.TLabel").pack(anchor="w", pady=(8, 3))
            for tool in tools:
                button = ttk.Button(
                    sidebar_content,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", pady=2)
                button.bind("<Enter>", lambda _event, selected=tool: self.status.set(selected.description))
                button.bind("<Leave>", lambda _event: self.status.set("Ready."))

        self.image_view = ImageView(self.body)
        self.body.add(self.image_view, weight=3)

        # Graph / Plot Side Panel (shown on demand when a graph/histogram is generated)
        self.graph_panel = ttk.Frame(self.body, padding=8, width=390)
        self.graph_panel.pack_propagate(False)

        graph_header = ttk.Frame(self.graph_panel)
        graph_header.pack(fill="x", pady=(0, 4))
        self.graph_title = ttk.Label(graph_header, text="Graph / Analysis", style="Category.TLabel")
        self.graph_title.pack(side="left")
        ttk.Button(graph_header, text="✕ Close", width=7, command=self.hide_graph_panel).pack(side="right")

        # Interactive channel selection filter toolbar for RGB Histograms
        self.channel_filter_frame = ttk.Frame(self.graph_panel)
        ttk.Label(self.channel_filter_frame, text="Channels:", font=("TkDefaultFont", 8, "bold")).pack(side="left", padx=(0, 4))
        self.hist_r_var = tk.BooleanVar(value=True)
        self.hist_g_var = tk.BooleanVar(value=True)
        self.hist_b_var = tk.BooleanVar(value=True)

        self.r_cb = ttk.Checkbutton(self.channel_filter_frame, text="Red", variable=self.hist_r_var, command=self._on_hist_channel_toggle)
        self.r_cb.pack(side="left", padx=2)
        self.g_cb = ttk.Checkbutton(self.channel_filter_frame, text="Green", variable=self.hist_g_var, command=self._on_hist_channel_toggle)
        self.g_cb.pack(side="left", padx=2)
        self.b_cb = ttk.Checkbutton(self.channel_filter_frame, text="Blue", variable=self.hist_b_var, command=self._on_hist_channel_toggle)
        self.b_cb.pack(side="left", padx=2)

        self._active_hist_img_np: np.ndarray | None = None

        self.graph_view = ImageView(self.graph_panel, default_filename="graph_plot.png")
        self.graph_view.pack(fill="both", expand=True)
        self._graph_panel_visible = False

        inspector = ttk.Frame(self.body, padding=12, width=240)
        inspector.pack_propagate(False)
        self.body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree headings", height=15)
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def show_graph(
        self,
        graph_image: Image.Image,
        title: str = "Graph / Analysis",
        is_rgb_hist: bool = False,
        hist_img_np: np.ndarray | None = None,
    ) -> None:
        """Display a graph/visualization in the side panel without affecting current image."""
        self.graph_title.configure(text=title)
        self.graph_view.show(graph_image)
        self.graph_view.set_default_filename(f"{title.lower().replace(' ', '_')}.png")

        if is_rgb_hist and hist_img_np is not None:
            self._active_hist_img_np = hist_img_np
            self.hist_r_var.set(True)
            self.hist_g_var.set(True)
            self.hist_b_var.set(True)
            self.channel_filter_frame.pack(after=self.graph_title.master, fill="x", pady=(2, 4))
        else:
            self._active_hist_img_np = None
            self.channel_filter_frame.pack_forget()

        if not self._graph_panel_visible:
            # Insert graph panel before inspector
            self.body.insert(2, self.graph_panel, weight=2)
            self._graph_panel_visible = True

    def _on_hist_channel_toggle(self) -> None:
        if self._active_hist_img_np is None:
            return
        from forensics_app.tools.histogram import plot_skimage_histogram
        stem = self.document.path.stem if self.document.path else "Image"
        updated_fig = plot_skimage_histogram(
            self._active_hist_img_np,
            show_r=self.hist_r_var.get(),
            show_g=self.hist_g_var.get(),
            show_b=self.hist_b_var.get(),
            title=f"{stem} (RGB Histogram)",
        )
        self.graph_view.show(updated_fig)

    def hide_graph_panel(self) -> None:
        """Hide the graph side panel."""
        if self._graph_panel_visible:
            self.body.forget(self.graph_panel)
            self._graph_panel_visible = False

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        if result is None:
            self.status.set(f"Cancelled {tool.title}.")
            return
        if result.image is not None:
            self.document.apply(result.image)
        if result.graph is not None:
            is_hist = (tool.tool_id == "histogram")
            is_rgb = False
            hist_np = None
            if is_hist and self.document.current is not None:
                is_rgb = self.document.current.mode not in ("L", "1")
                if is_rgb:
                    hist_np = np.array(self.document.current.convert("RGB"))
            self.show_graph(
                result.graph,
                title=f"{tool.title} Graph",
                is_rgb_hist=(is_hist and is_rgb),
                hist_img_np=hist_np,
            )
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self.image_view.show(self.document.current)
        if self.document.path:
            self.image_view.set_default_filename(f"{self.document.path.stem}_result.png")
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
