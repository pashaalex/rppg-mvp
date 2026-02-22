import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk  # pip install pillow


ECG_DIR = Path("ECG")
MARK_CSV = Path("mark.csv")
PNG_GLOB = "*.png"


class RMarkerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECG R-marker")
        self.geometry("1100x700")

        self.records = []  # list of dict: {"file": "frame_00001.png", "mark": 0/1}
        self.photo = None  # keep reference to prevent GC

        self._build_ui()
        self._load_or_init_data()
        self._populate_tree()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Select first item by default
        if self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
            self._on_select(None)

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # Left: image
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.img_label = ttk.Label(left, anchor="center")
        self.img_label.grid(row=0, column=0, sticky="nsew")

        # Right: list
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            right,
            columns=("mark", "file"),
            show="headings",
            selectmode="browse",
            height=20
        )
        self.tree.heading("mark", text="Mark")
        self.tree.heading("file", text="File")
        self.tree.column("mark", width=60, anchor="center", stretch=False)
        self.tree.column("file", width=400, anchor="w", stretch=True)

        vsb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Hint
        hint = ttk.Label(
            right,
            text="↑/↓: navigate   Space/Enter: toggle '+' (R peak)",
            anchor="w"
        )
        hint.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # Bindings
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.bind("<Up>", self._on_up_down)
        self.bind("<Down>", self._on_up_down)
        self.bind("<Return>", self._toggle_mark)
        self.bind("<space>", self._toggle_mark)
        self.tree.bind("<Double-1>", self._on_double_click)

    def _load_or_init_data(self):
        if MARK_CSV.exists():
            self.records = self._read_csv(MARK_CSV)
            if not self.records:
                messagebox.showwarning("Warning", "mark.csv exists but is empty/invalid. Reinitializing from ECG folder.")
                self.records = self._scan_folder()
                self._write_csv(MARK_CSV, self.records)
        else:
            self.records = self._scan_folder()
            self._write_csv(MARK_CSV, self.records)

    def _scan_folder(self):
        if not ECG_DIR.exists():
            messagebox.showerror("Error", f"Folder '{ECG_DIR}' not found.")
            return []

        files = sorted([p.name for p in ECG_DIR.glob(PNG_GLOB)])
        if not files:
            messagebox.showerror("Error", f"No PNG files found in '{ECG_DIR}'.")
            return []

        return [{"file": fn, "mark": 0} for fn in files]

    def _read_csv(self, path: Path):
        out = []
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    # support optional header
                    if row[0].strip().lower() in ("file", "filename") and len(row) >= 2:
                        continue
                    fn = row[0].strip()
                    mk = row[1].strip() if len(row) >= 2 else "0"
                    out.append({"file": fn, "mark": 1 if mk in ("1", "+", "true", "True") else 0})
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read {path}: {e}")
            return []
        return out

    def _write_csv(self, path: Path, records):
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "mark"])
                for r in records:
                    writer.writerow([r["file"], int(r["mark"])])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write {path}: {e}")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.records):
            mark_txt = "+" if r["mark"] else ""
            self.tree.insert("", "end", iid=str(i), values=(mark_txt, r["file"]))

    def _get_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _on_select(self, _evt):
        idx = self._get_selected_index()
        if idx is None:
            return
        fn = self.records[idx]["file"]
        img_path = ECG_DIR / fn
        if not img_path.exists():
            self.img_label.config(text=f"Missing: {img_path}", image="")
            self.photo = None
            return

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img2 = img.resize((w * 2, h * 2), Image.NEAREST)
            self.photo = ImageTk.PhotoImage(img2)
            self.img_label.config(image=self.photo, text="")
        except Exception as e:
            self.img_label.config(text=f"Failed to load image: {e}", image="")
            self.photo = None

    def _on_up_down(self, evt):
        items = self.tree.get_children()
        if not items:
            return "break"

        idx = self._get_selected_index()
        if idx is None:
            idx = 0
        else:
            if evt.keysym == "Up":
                idx = max(0, idx - 1)
            elif evt.keysym == "Down":
                idx = min(len(items) - 1, idx + 1)

        iid = str(idx)
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        self.tree.see(iid)
        self._on_select(None)
        return "break"

    def _toggle_mark(self, _evt=None):
        idx = self._get_selected_index()
        if idx is None:
            return "break"

        self.records[idx]["mark"] = 0 if self.records[idx]["mark"] else 1
        mark_txt = "+" if self.records[idx]["mark"] else ""
        self.tree.set(str(idx), "mark", mark_txt)
        return "break"

    def _on_double_click(self, evt):
        # Double-click toggles mark (anywhere on row)
        self._toggle_mark(evt)

    def on_close(self):
        self._write_csv(MARK_CSV, self.records)
        self.destroy()


if __name__ == "__main__":
    app = RMarkerApp()
    app.mainloop()
