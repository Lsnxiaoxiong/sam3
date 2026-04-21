from __future__ import annotations

import os
import pathlib
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import PIL.Image
import PIL.ImageTk

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from onnx_demo_lib import InferenceResult, Sam3OnnxDemo


CANVAS_SIZE = (560, 560)
POINT_RADIUS = 5


def _default_model_dir() -> str:
    output_dir = (CURRENT_DIR.parent / "output").resolve()
    if output_dir.exists():
        return str(output_dir)
    return ""


def _default_image_path(*parts: str) -> str:
    path = (CURRENT_DIR.parent / pathlib.Path(*parts)).resolve()
    return str(path) if path.exists() else ""


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.h_scrollbar = ttk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview
        )
        self.content = ttk.Frame(self.canvas)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self._window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._window, width=event.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event: tk.Event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        current = widget
        while current is not None:
            if current == self:
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
                return
            current = current.master


class ResultPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master)
        ttk.Label(self, text=title).pack(anchor="w")
        self._canvas = tk.Canvas(
            self,
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
            bg="#101010",
            highlightthickness=1,
            highlightbackground="#404040",
        )
        self._canvas.pack(fill="both", expand=True)
        self._photo: PIL.ImageTk.PhotoImage | None = None
        self._summary = tk.Text(self, height=9)
        self._summary.pack(fill="x", pady=(8, 0))
        self.set_result(None)

    def set_result(self, result: InferenceResult | None) -> None:
        self._canvas.delete("all")
        if result is None:
            self._canvas.create_text(
                CANVAS_SIZE[0] // 2,
                CANVAS_SIZE[1] // 2,
                text="No result",
                fill="white",
            )
            self._photo = None
            self._summary.delete("1.0", "end")
            return
        image = result.overlay.copy()
        image.thumbnail(CANVAS_SIZE, PIL.Image.Resampling.LANCZOS)
        self._photo = PIL.ImageTk.PhotoImage(image)
        self._canvas.create_image(
            CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2, image=self._photo
        )
        self._summary.delete("1.0", "end")
        self._summary.insert("1.0", result.summary)


@dataclass
class PointPrompt:
    x: float
    y: float
    label: int


class AnnotationCanvas(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        *,
        allow_points: bool,
        allow_box: bool,
    ) -> None:
        super().__init__(master)
        ttk.Label(self, text=title).pack(anchor="w")
        self.canvas = tk.Canvas(
            self,
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
            bg="#101010",
            highlightthickness=1,
            highlightbackground="#404040",
        )
        self.canvas.pack(fill="both", expand=True)
        self.allow_points = allow_points
        self.allow_box = allow_box
        self.image: PIL.Image.Image | None = None
        self._display_photo: PIL.ImageTk.PhotoImage | None = None
        self._display_size = (1, 1)
        self._offset = (0, 0)
        self.points: list[PointPrompt] = []
        self.box_xyxy: tuple[float, float, float, float] | None = None
        self.tool_var = tk.StringVar(
            value="positive" if allow_points else "box" if allow_box else "view"
        )
        self._drag_start: tuple[float, float] | None = None
        self._drag_preview: int | None = None
        self.path_var = tk.StringVar()

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Label(controls, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.path_var, width=64).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Browse", command=self._browse_image).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(controls, text="Load", command=self.load_from_path).grid(
            row=0, column=3, padx=4
        )
        controls.columnconfigure(1, weight=1)

        if allow_points or allow_box:
            tools = ttk.LabelFrame(self, text="Annotation Tools", padding=8)
            tools.pack(fill="x", pady=(8, 0))
            column = 0
            if allow_points:
                ttk.Radiobutton(
                    tools, text="Positive Point", value="positive", variable=self.tool_var
                ).grid(row=0, column=column, sticky="w")
                column += 1
                ttk.Radiobutton(
                    tools, text="Negative Point", value="negative", variable=self.tool_var
                ).grid(row=0, column=column, sticky="w", padx=(12, 0))
                column += 1
            if allow_box:
                ttk.Radiobutton(
                    tools, text="Box", value="box", variable=self.tool_var
                ).grid(row=0, column=column, sticky="w", padx=(12, 0))
                column += 1
            ttk.Button(tools, text="Clear Points", command=self.clear_points).grid(
                row=1, column=0, sticky="w", pady=(8, 0)
            )
            ttk.Button(tools, text="Clear Box", command=self.clear_box).grid(
                row=1, column=1, sticky="w", padx=(12, 0), pady=(8, 0)
            )

        self.info_var = tk.StringVar(value="No image loaded")
        ttk.Label(self, textvariable=self.info_var).pack(anchor="w", pady=(8, 0))

        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self._draw()

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.path_var.set(path)
            self.load_from_path()

    def load_from_path(self) -> None:
        raw_path = self.path_var.get().strip()
        if not raw_path:
            return
        path = pathlib.Path(raw_path)
        with PIL.Image.open(path) as image:
            self.image = image.convert("RGB")
        self.points = []
        self.box_xyxy = None
        self._drag_start = None
        self._draw()

    def set_path(self, path: str) -> None:
        self.path_var.set(path)
        if path:
            self.load_from_path()

    def clear_points(self) -> None:
        self.points = []
        self._draw()

    def clear_box(self) -> None:
        self.box_xyxy = None
        self._drag_start = None
        self._draw()

    def point_coords_text(self) -> str | None:
        if not self.points:
            return None
        return ";".join(f"{point.x:.1f},{point.y:.1f}" for point in self.points)

    def point_labels_text(self) -> str | None:
        if not self.points:
            return None
        return ",".join(str(point.label) for point in self.points)

    def box_text(self) -> str | None:
        if self.box_xyxy is None:
            return None
        x0, y0, x1, y1 = self.box_xyxy
        return f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y = self._offset
        display_w, display_h = self._display_size
        if self.image is None:
            return x, y
        return (
            offset_x + x * display_w / self.image.width,
            offset_y + y * display_h / self.image.height,
        )

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.image is None:
            return None
        offset_x, offset_y = self._offset
        display_w, display_h = self._display_size
        if (
            x < offset_x
            or y < offset_y
            or x > offset_x + display_w
            or y > offset_y + display_h
        ):
            return None
        img_x = (x - offset_x) * self.image.width / display_w
        img_y = (y - offset_y) * self.image.height / display_h
        img_x = max(0.0, min(float(self.image.width - 1), img_x))
        img_y = max(0.0, min(float(self.image.height - 1), img_y))
        return img_x, img_y

    def _on_left_down(self, event: tk.Event) -> None:
        image_xy = self._canvas_to_image(event.x, event.y)
        if image_xy is None:
            return
        tool = self.tool_var.get()
        if tool == "positive" and self.allow_points:
            self.points.append(PointPrompt(image_xy[0], image_xy[1], 1))
            self._draw()
            return
        if tool == "negative" and self.allow_points:
            self.points.append(PointPrompt(image_xy[0], image_xy[1], 0))
            self._draw()
            return
        if tool == "box" and self.allow_box:
            self._drag_start = image_xy
            self.box_xyxy = (image_xy[0], image_xy[1], image_xy[0], image_xy[1])
            self._draw()

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self.tool_var.get() != "box":
            return
        image_xy = self._canvas_to_image(event.x, event.y)
        if image_xy is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = image_xy
        self.box_xyxy = (
            min(x0, x1),
            min(y0, y1),
            max(x0, x1),
            max(y0, y1),
        )
        self._draw()

    def _on_left_up(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        self._drag_start = None
        if self.box_xyxy is not None:
            x0, y0, x1, y1 = self.box_xyxy
            if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
                self.box_xyxy = None
        self._draw()

    def _on_right_click(self, event: tk.Event) -> None:
        image_xy = self._canvas_to_image(event.x, event.y)
        if image_xy is None:
            return
        if self.allow_points and self.points:
            nearest_index = min(
                range(len(self.points)),
                key=lambda idx: (self.points[idx].x - image_xy[0]) ** 2
                + (self.points[idx].y - image_xy[1]) ** 2,
            )
            point = self.points[nearest_index]
            if abs(point.x - image_xy[0]) < 20 and abs(point.y - image_xy[1]) < 20:
                del self.points[nearest_index]
                self._draw()
                return
        if self.allow_box and self.box_xyxy is not None:
            self.box_xyxy = None
            self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(
                CANVAS_SIZE[0] // 2,
                CANVAS_SIZE[1] // 2,
                text="Load an image to start",
                fill="white",
            )
            self.info_var.set("No image loaded")
            return

        display_image = self.image.copy()
        display_image.thumbnail(CANVAS_SIZE, PIL.Image.Resampling.LANCZOS)
        self._display_size = display_image.size
        self._offset = (
            (CANVAS_SIZE[0] - display_image.size[0]) // 2,
            (CANVAS_SIZE[1] - display_image.size[1]) // 2,
        )
        self._display_photo = PIL.ImageTk.PhotoImage(display_image)
        self.canvas.create_image(
            self._offset[0], self._offset[1], image=self._display_photo, anchor="nw"
        )

        for point in self.points:
            cx, cy = self._image_to_canvas(point.x, point.y)
            color = "#32cd32" if point.label == 1 else "#ff4040"
            self.canvas.create_oval(
                cx - POINT_RADIUS,
                cy - POINT_RADIUS,
                cx + POINT_RADIUS,
                cy + POINT_RADIUS,
                fill=color,
                outline="white",
                width=1,
            )

        if self.box_xyxy is not None:
            x0, y0, x1, y1 = self.box_xyxy
            left, top = self._image_to_canvas(x0, y0)
            right, bottom = self._image_to_canvas(x1, y1)
            self.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                outline="#00bfff",
                width=2,
            )

        info = [
            f"size: {self.image.width}x{self.image.height}",
            f"points: {len(self.points)}",
            f"box: {'yes' if self.box_xyxy is not None else 'no'}",
        ]
        self.info_var.set(" | ".join(info))


class CrossImageTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=8)
        self.reference_canvas = AnnotationCanvas(
            self, "Reference Image", allow_points=False, allow_box=True
        )
        self.target_canvas = AnnotationCanvas(
            self, "Current Target Image", allow_points=False, allow_box=False
        )
        self.result_panel = ResultPanel(self, "Result")
        self.results: list[InferenceResult] = []

        controls = ttk.LabelFrame(self, text="Cross-image Settings", padding=8)
        controls.pack(fill="x")
        self.prompt_var = tk.StringVar(value="white T-shirt")
        ttk.Label(controls, text="Text Prompt").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.prompt_var, width=40).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        controls.columnconfigure(1, weight=1)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(8, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        self.reference_canvas.grid(in_=body, row=0, column=0, sticky="nsew")
        self.target_canvas.grid(in_=body, row=0, column=1, sticky="nsew", padx=(8, 0))

        right = ttk.Frame(body)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.result_panel.pack(in_=right, fill="both", expand=True)
        ttk.Label(right, text="Results").pack(anchor="w", pady=(8, 0))
        self.result_list = tk.Listbox(right, height=8)
        self.result_list.pack(fill="x")
        self.result_list.bind("<<ListboxSelect>>", self._on_select_result)

        bottom = ttk.LabelFrame(self, text="Target Images", padding=8)
        bottom.pack(fill="x", pady=(8, 0))
        self.target_path_var = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.target_path_var, width=60).pack(
            fill="x"
        )
        buttons = ttk.Frame(bottom)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Add", command=self._add_target).pack(side="left")
        ttk.Button(buttons, text="Browse", command=self._browse_target).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(buttons, text="Remove", command=self._remove_target).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(
            buttons, text="Run Cross-image Inference", command=self._run
        ).pack(side="right")
        self.target_list = tk.Listbox(bottom, height=6)
        self.target_list.pack(fill="x", expand=True, pady=(8, 0))
        self.target_list.bind("<<ListboxSelect>>", self._on_select_target)

        self.reference_canvas.set_path(_default_image_path("assets", "videos", "0001", "9.jpg"))
        self.target_canvas.set_path(_default_image_path("assets", "videos", "0001", "14.jpg"))
        for relative in ("14.jpg", "20.jpg"):
            target = _default_image_path("assets", "videos", "0001", relative)
            if target:
                self.target_list.insert("end", target)
        if self.target_list.size():
            self.target_list.selection_set(0)
            self._on_select_target()

    def _browse_target(self) -> None:
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        for path in paths:
            self.target_list.insert("end", path)
        if paths:
            self.target_list.selection_clear(0, "end")
            self.target_list.selection_set("end")
            self._on_select_target()

    def _add_target(self) -> None:
        path = self.target_path_var.get().strip()
        if path:
            self.target_list.insert("end", path)
            self.target_path_var.set("")

    def _remove_target(self) -> None:
        indexes = list(self.target_list.curselection())
        indexes.reverse()
        for index in indexes:
            self.target_list.delete(index)

    def _on_select_target(self, _event=None) -> None:
        selection = self.target_list.curselection()
        if not selection:
            return
        self.target_canvas.set_path(self.target_list.get(selection[0]))

    def _run(self) -> None:
        app = self.winfo_toplevel()
        if not hasattr(app, "run_cross_image"):
            return
        app.run_cross_image()

    def _on_select_result(self, _event=None) -> None:
        selection = self.result_list.curselection()
        if not selection:
            return
        self.result_panel.set_result(self.results[selection[0]])


class Sam3OnnxGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SAM3 ONNX Demo")
        self.geometry("1820x1040")
        self.demo = Sam3OnnxDemo(_default_model_dir() or None)
        self.status_var = tk.StringVar(value="Ready")
        self.model_dir_var = tk.StringVar(value=_default_model_dir())
        self.score_threshold_var = tk.StringVar(value="0.5")
        self.model_state_var = tk.StringVar(value=self.demo.loaded_summary())
        self.provider_var = tk.StringVar(value="cuda")
        self.low_vram_var = tk.BooleanVar(value=True)
        self.gpu_mem_limit_var = tk.StringVar(value="")
        os.environ.setdefault("SAM3_ORT_PROVIDER", "cuda")
        os.environ.setdefault("SAM3_ORT_CUDNN_MAX_WORKSPACE", "0")
        os.environ.setdefault("SAM3_ORT_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.LabelFrame(root, text="Global Settings", padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Model Dir").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_dir_var, width=80).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(top, text="Browse", command=self._choose_model_dir).grid(
            row=0, column=2, padx=4
        )
        ttk.Label(top, text="Score Threshold").grid(row=0, column=3, sticky="w")
        ttk.Entry(top, textvariable=self.score_threshold_var, width=8).grid(
            row=0, column=4, sticky="w", padx=4
        )
        ttk.Label(top, text="Provider").grid(row=0, column=5, sticky="w", padx=(12, 0))
        ttk.Combobox(
            top,
            textvariable=self.provider_var,
            values=("cuda", "cpu"),
            state="readonly",
            width=8,
        ).grid(row=0, column=6, sticky="w")
        ttk.Label(top, text="GPU Limit MB").grid(
            row=0, column=7, sticky="w", padx=(12, 0)
        )
        ttk.Entry(top, textvariable=self.gpu_mem_limit_var, width=10).grid(
            row=0, column=8, sticky="w"
        )
        ttk.Checkbutton(top, text="Low VRAM", variable=self.low_vram_var).grid(
            row=0, column=9, sticky="w", padx=(12, 0)
        )
        ttk.Button(top, text="Apply Runtime", command=self._apply_runtime_settings).grid(
            row=0, column=10, sticky="w", padx=(8, 0)
        )
        ttk.Button(top, text="Load Point/Box", command=self._load_interactive).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Unload Point/Box", command=self._unload_interactive).grid(
            row=1, column=1, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Load Text", command=self._load_grounding).grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Unload Text", command=self._unload_grounding).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Load Cross-image", command=self._load_cross_image).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Unload Cross-image", command=self._unload_cross_image).grid(
            row=2, column=1, sticky="w", pady=(8, 0)
        )
        ttk.Button(top, text="Unload All", command=self._unload_all).grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(top, text="Model State").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(top, textvariable=self.model_state_var).grid(
            row=3, column=1, columnspan=4, sticky="w", pady=(8, 0)
        )
        top.columnconfigure(1, weight=1)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=(12, 8))

        self.interactive_tab = ScrollableFrame(notebook)
        notebook.add(self.interactive_tab, text="Point / Box Prompt")
        self._build_interactive_tab(self.interactive_tab.content)

        self.grounding_tab = ScrollableFrame(notebook)
        notebook.add(self.grounding_tab, text="Text Prompt")
        self._build_grounding_tab(self.grounding_tab.content)

        self.cross_tab_scroll = ScrollableFrame(notebook)
        self.cross_tab = CrossImageTab(self.cross_tab_scroll.content)
        self.cross_tab.pack(fill="both", expand=True)
        notebook.add(self.cross_tab_scroll, text="Cross-image Reference")

        ttk.Label(root, textvariable=self.status_var).pack(anchor="w")

    def _build_interactive_tab(self, parent: ttk.Frame) -> None:
        self.interactive_canvas = AnnotationCanvas(
            parent, "Preview Image", allow_points=True, allow_box=True
        )
        self.interactive_canvas.pack(side="left", fill="both", expand=True)
        self.interactive_canvas.set_path(_default_image_path("assets", "images", "groceries.jpg"))

        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        controls = ttk.LabelFrame(right, text="Inference Options", padding=8)
        controls.pack(fill="x")
        self.interactive_multimask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls, text="Multimask Output", variable=self.interactive_multimask_var
        ).pack(anchor="w")
        ttk.Button(
            controls, text="Run Interactive Inference", command=self._run_interactive
        ).pack(fill="x", pady=(8, 0))
        self.interactive_result_panel = ResultPanel(right, "Inference Result")
        self.interactive_result_panel.pack(fill="both", expand=True, pady=(8, 0))

    def _build_grounding_tab(self, parent: ttk.Frame) -> None:
        self.grounding_canvas = AnnotationCanvas(
            parent, "Preview Image", allow_points=False, allow_box=True
        )
        self.grounding_canvas.pack(side="left", fill="both", expand=True)
        self.grounding_canvas.set_path(_default_image_path("assets", "images", "groceries.jpg"))

        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        controls = ttk.LabelFrame(right, text="Grounding Options", padding=8)
        controls.pack(fill="x")
        self.grounding_prompt_var = tk.StringVar(value="red light")
        ttk.Label(controls, text="Text Prompt").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.grounding_prompt_var).pack(fill="x")
        ttk.Button(
            controls, text="Run Text Inference", command=self._run_grounding
        ).pack(fill="x", pady=(8, 0))
        self.grounding_result_panel = ResultPanel(right, "Inference Result")
        self.grounding_result_panel.pack(fill="both", expand=True, pady=(8, 0))

    def _choose_model_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.model_dir_var.set(path)
            self._refresh_demo()

    def _refresh_demo(self) -> None:
        model_dir = self.model_dir_var.get().strip()
        self.demo.set_model_dir(model_dir or None)
        self._sync_model_state()

    def _apply_runtime_settings(self) -> None:
        provider = self.provider_var.get().strip().lower()
        os.environ["SAM3_ORT_PROVIDER"] = provider

        gpu_limit = self.gpu_mem_limit_var.get().strip()
        if gpu_limit:
            os.environ["SAM3_ORT_GPU_MEM_LIMIT_MB"] = gpu_limit
        else:
            os.environ.pop("SAM3_ORT_GPU_MEM_LIMIT_MB", None)

        if self.low_vram_var.get():
            os.environ["SAM3_ORT_CUDNN_MAX_WORKSPACE"] = "0"
            os.environ["SAM3_ORT_CUDNN_CONV_ALGO_SEARCH"] = "HEURISTIC"
        else:
            os.environ.pop("SAM3_ORT_CUDNN_MAX_WORKSPACE", None)
            os.environ["SAM3_ORT_CUDNN_CONV_ALGO_SEARCH"] = "DEFAULT"

        self.demo.unload_all()
        self._refresh_demo()
        self.status_var.set("Runtime settings applied. Reload models to take effect.")

    def _sync_model_state(self) -> None:
        self.model_state_var.set(self.demo.loaded_summary())

    def _load_interactive(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_interactive()
            self.after(0, self._sync_model_state)

        self._run_async("Loading point/box model...", task)

    def _unload_interactive(self) -> None:
        self.demo.unload_interactive()
        self._sync_model_state()
        self.status_var.set("Point/box model unloaded")

    def _load_grounding(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_grounding()
            self.after(0, self._sync_model_state)

        self._run_async("Loading text model...", task)

    def _unload_grounding(self) -> None:
        self.demo.unload_grounding()
        self._sync_model_state()
        self.status_var.set("Text model unloaded")

    def _load_cross_image(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_cross_image()
            self.after(0, self._sync_model_state)

        self._run_async("Loading cross-image model...", task)

    def _unload_cross_image(self) -> None:
        self.demo.unload_cross_image()
        self._sync_model_state()
        self.status_var.set("Cross-image model unloaded")

    def _unload_all(self) -> None:
        self.demo.unload_all()
        self._sync_model_state()
        self.status_var.set("All models unloaded")

    def _score_threshold(self) -> float:
        return float(self.score_threshold_var.get().strip() or "0.5")

    def _run_async(self, status: str, callback) -> None:
        self.status_var.set(status)

        def worker() -> None:
            try:
                callback()
                self.after(0, lambda: self.status_var.set("Ready"))
            except Exception as exc:
                self.after(
                    0,
                    lambda: (
                        self.status_var.set("Failed"),
                        messagebox.showerror("SAM3 ONNX Demo", str(exc)),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _run_interactive(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_interactive()
            image_path = self.interactive_canvas.path_var.get().strip()
            result = self.demo.point_or_box_prompt(
                image_path=image_path,
                point_coords=self.interactive_canvas.point_coords_text(),
                point_labels=self.interactive_canvas.point_labels_text(),
                box_prompt=self.interactive_canvas.box_text(),
                multimask_output=self.interactive_multimask_var.get(),
            )
            self.after(
                0,
                lambda: (
                    self.interactive_result_panel.set_result(result),
                    self._sync_model_state(),
                ),
            )

        self._run_async("Running interactive inference...", task)

    def _run_grounding(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_grounding()
            image_path = self.grounding_canvas.path_var.get().strip()
            result = self.demo.text_prompt(
                image_path=image_path,
                text_prompt=self.grounding_prompt_var.get().strip(),
                grounding_boxes=self.grounding_canvas.box_text(),
                grounding_box_labels="1" if self.grounding_canvas.box_text() else None,
                score_threshold=self._score_threshold(),
            )
            self.after(
                0,
                lambda: (
                    self.grounding_result_panel.set_result(result),
                    self._sync_model_state(),
                ),
            )

        self._run_async("Running text prompt inference...", task)

    def run_cross_image(self) -> None:
        def task() -> None:
            self._refresh_demo()
            self.demo.load_cross_image()
            reference_image = self.cross_tab.reference_canvas.path_var.get().strip()
            reference_boxes = self.cross_tab.reference_canvas.box_text()
            target_images = list(self.cross_tab.target_list.get(0, "end"))
            if not reference_boxes:
                raise ValueError("Draw a reference box on the reference image first.")
            results = self.demo.cross_image_box_transfer(
                reference_image=reference_image,
                reference_boxes=reference_boxes,
                target_images=target_images,
                text_prompt=self.cross_tab.prompt_var.get().strip(),
                score_threshold=self._score_threshold(),
            )

            def update_ui() -> None:
                self.cross_tab.results = results
                self.cross_tab.result_list.delete(0, "end")
                for result in results:
                    self.cross_tab.result_list.insert("end", str(result.image_path))
                if results:
                    self.cross_tab.result_list.selection_set(0)
                    self.cross_tab.result_panel.set_result(results[0])
                self._sync_model_state()

            self.after(0, update_ui)

        self._run_async("Running cross-image reference inference...", task)

    def _on_close(self) -> None:
        self.demo.unload_all()
        self.destroy()


def main() -> None:
    app = Sam3OnnxGui()
    app.mainloop()


if __name__ == "__main__":
    main()
