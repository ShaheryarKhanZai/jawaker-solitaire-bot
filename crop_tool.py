"""
Screen Annotator — Python/Tkinter GUI
--------------------------------------
Flow:
  1. Takes a screenshot (2s delay so you can switch windows)
  2. Crops min→max of all normalized coordinates in REGIONS
  3. Resizes crop to 461×580
  4. Opens GUI — draw boxes, label them, save crops

Keyboard shortcuts:
  B / F1           Box tool
  V / F2           Select/Move tool
  Delete           Delete selected box
  Ctrl+Z           Undo last box
  Ctrl+S           Save selected crop
  Ctrl+Shift+S     Save all crops
  Escape           Deselect

Install:
  pip install pillow pyautogui
  # Linux only: sudo apt-get install python3-tk scrot
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import pyautogui
import os, time, json

# ── Config ────────────────────────────────────────────────────────────────────

REGIONS = {
    "f0":    {"x1": 0.3749748547871908,  "y1": 0.21142438252766926, "x2": 0.4079789956410726,  "y2": 0.2962740862811053},
    "f1":    {"x1": 0.4079789956410726,  "y1": 0.21142438252766926, "x2": 0.4409831364949544,  "y2": 0.2962740862811053},
    "f2":    {"x1": 0.4409831364949544,  "y1": 0.21142438252766926, "x2": 0.4739872773488363,  "y2": 0.2962740862811053},
    "f3":    {"x1": 0.4739872773488363,  "y1": 0.21142438252766926, "x2": 0.5069914182027181,  "y2": 0.2962740862811053},
    "t0":    {"x1": 0.3749748547871908,  "y1": 0.29950900607638886, "x2": 0.40777945745558963, "y2": 0.7481481481481481},
    "t1":    {"x1": 0.40777945745558963, "y1": 0.29950900607638886, "x2": 0.44058406012398854, "y2": 0.7481481481481481},
    "t2":    {"x1": 0.4405840601239886,  "y1": 0.29950900607638886, "x2": 0.47338866279238745, "y2": 0.7481481481481481},
    "t3":    {"x1": 0.47338866279238745, "y1": 0.29950900607638886, "x2": 0.5061932654607864,  "y2": 0.7481481481481481},
    "t4":    {"x1": 0.5061932654607864,  "y1": 0.29950900607638886, "x2": 0.5389978681291853,  "y2": 0.7481481481481481},
    "t5":    {"x1": 0.5389978681291853,  "y1": 0.29950900607638886, "x2": 0.5718024707975842,  "y2": 0.7481481481481481},
    "t6":    {"x1": 0.5718024707975842,  "y1": 0.29950900607638886, "x2": 0.6046070734659831,  "y2": 0.7481481481481481},
    "stock": {"x1": 0.572889773050944,   "y1": 0.21360181172688802, "x2": 0.6046070734659831,  "y2": 0.29756206936306423},
    "waste": {"x1": 0.5415225982666015,  "y1": 0.21360181172688802, "x2": 0.5732398986816406,  "y2": 0.29799711439344617},
}

CANVAS_W, CANVAS_H = 461, 580
OUTPUT_DIR = "annotated_crops"

PALETTE  = ["#E24B4A", "#378ADD", "#1D9E75", "#EF9F27", "#D4537E", "#7F77DD", "#FFFFFF"]
BG       = "#1c1c1e"
BG2      = "#2c2c2e"
BG3      = "#3a3a3c"
FG       = "#f5f5f7"
FG2      = "#aeaeb2"
ACCENT   = "#0a84ff"
DANGER   = "#ff453a"
SUCCESS  = "#30d158"

def _font(size=10, bold=False):
    for name in ("Segoe UI", "SF Pro Display", "Helvetica Neue", "Helvetica"):
        try:
            import tkinter.font as tkf
            if name in tkf.families():
                return (name, size, "bold") if bold else (name, size)
        except Exception:
            pass
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)

# ── Screenshot helpers ────────────────────────────────────────────────────────

def get_crop_bounds():
    x1 = min(r["x1"] for r in REGIONS.values())
    y1 = min(r["y1"] for r in REGIONS.values())
    x2 = max(r["x2"] for r in REGIONS.values())
    y2 = max(r["y2"] for r in REGIONS.values())
    return x1, y1, x2, y2

def take_screenshot():
    time.sleep(2)
    return pyautogui.screenshot()

def crop_and_resize(screen):
    sw, sh = screen.size
    nx1, ny1, nx2, ny2 = get_crop_bounds()
    cropped = screen.crop((int(nx1*sw), int(ny1*sh), int(nx2*sw), int(ny2*sh)))
    return cropped.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

# ── Annotation data ───────────────────────────────────────────────────────────

class Annotation:
    _ctr = 0
    def __init__(self, x1, y1, x2, y2, color="#E24B4A", label=""):
        Annotation._ctr += 1
        self.id              = Annotation._ctr
        self.x1, self.y1    = x1, y1
        self.x2, self.y2    = x2, y2
        self.color           = color
        self.label           = label
        self.rect_id         = None
        self.text_id         = None

    @property
    def w(self): return int(self.x2 - self.x1)
    @property
    def h(self): return int(self.y2 - self.y1)

    def contains(self, px, py):
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

# ── Main GUI ──────────────────────────────────────────────────────────────────

class AnnotatorApp:
    def __init__(self, root, image):
        self.root  = root
        self.image = image
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.annotations = []
        self.selected    = None
        self.tool        = "box"
        self.color       = PALETTE[0]
        self.drawing     = False
        self.drag_data   = {}
        self.start_xy    = (0, 0)
        self.live_rect   = None

        self._build_ui()
        self._bind_keys()
        self._set_status("Ready — draw a bounding box")

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Screen Annotator")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # Toolbar
        tb = tk.Frame(self.root, bg=BG2, pady=4)
        tb.pack(fill=tk.X)

        self.btn_box  = self._mk_btn(tb, "Box [B]",     lambda: self._set_tool("box"),  accent=True)
        self.btn_move = self._mk_btn(tb, "Select [V]",  lambda: self._set_tool("move"), accent=False)
        self.btn_box.pack(side=tk.LEFT, padx=(8, 2), pady=2)
        self.btn_move.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(tb, bg=BG3, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        tk.Label(tb, text="Color:", font=_font(9), fg=FG2, bg=BG2).pack(side=tk.LEFT, padx=(0, 4))
        self.color_btns = []
        for c in PALETTE:
            b = tk.Button(tb, bg=c, width=2, relief=tk.FLAT, bd=0,
                          highlightthickness=2, highlightbackground=BG2,
                          cursor="hand2", command=lambda col=c: self._set_color(col))
            b.pack(side=tk.LEFT, padx=2)
            self.color_btns.append((c, b))
        self._highlight_color(self.color)

        tk.Frame(tb, bg=BG3, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        self._mk_btn(tb, "↩ Undo",     self._undo).pack(side=tk.LEFT, padx=2, pady=2)
        self._mk_btn(tb, "✕ Clear all", self._clear_all).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(tb, bg=BG3, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        self._mk_btn(tb, "📷 New screenshot", self._new_screenshot).pack(side=tk.LEFT, padx=2, pady=2)

        # Body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Canvas
        cf = tk.Frame(body, bg=BG3, bd=1)
        cf.pack(side=tk.LEFT, padx=(8, 4), pady=8)

        self.tk_img = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(cf, width=CANVAS_W, height=CANVAS_H,
                                bg="#000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          lambda e: self.coord_var.set(f"x={e.x}  y={e.y}"))
        self.canvas.bind("<Leave>",           lambda e: self.coord_var.set(""))

        # Sidebar
        sb = tk.Frame(body, bg=BG2, width=220)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        sb.pack_propagate(False)

        tk.Label(sb, text="Annotations", font=_font(10, bold=True), fg=FG, bg=BG2
                 ).pack(anchor=tk.W, padx=10, pady=(10, 4))

        lf = tk.Frame(sb, bg=BG2)
        lf.pack(fill=tk.BOTH, expand=True, padx=6)

        sb_scroll = tk.Scrollbar(lf, orient=tk.VERTICAL, bg=BG3, troughcolor=BG2)
        self.listbox = tk.Listbox(lf, font=_font(9), fg=FG, bg=BG3,
                                  selectbackground=ACCENT, selectforeground="#fff",
                                  activestyle="none", relief=tk.FLAT, bd=0,
                                  yscrollcommand=sb_scroll.set, highlightthickness=0)
        sb_scroll.config(command=self.listbox.yview)
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        tk.Frame(sb, bg=BG3, height=1).pack(fill=tk.X, padx=6, pady=6)

        tk.Label(sb, text="Label", font=_font(9), fg=FG2, bg=BG2
                 ).pack(anchor=tk.W, padx=10)
        self.label_var = tk.StringVar()
        self.label_entry = tk.Entry(sb, textvariable=self.label_var,
                                    font=_font(10), fg=FG, bg=BG3,
                                    insertbackground=FG, relief=tk.FLAT, bd=0,
                                    highlightthickness=1, highlightbackground=BG3,
                                    highlightcolor=ACCENT)
        self.label_entry.pack(fill=tk.X, padx=10, pady=(2, 8), ipady=5)
        self.label_entry.bind("<Return>", lambda e: self._save_selected())
        self.label_var.trace_add("write", lambda *_: self._update_label_live())

        self.save_btn = tk.Button(sb, text="💾  Save crop  [Ctrl+S]",
                                  font=_font(10, bold=True), fg="#fff", bg=ACCENT,
                                  activebackground="#0070d8", activeforeground="#fff",
                                  relief=tk.FLAT, bd=0, cursor="hand2", pady=7,
                                  command=self._save_selected)
        self.save_btn.pack(fill=tk.X, padx=10, pady=2)

        tk.Button(sb, text="💾  Save all  [Ctrl+Shift+S]",
                  font=_font(10), fg=FG, bg=BG3,
                  activebackground=BG3, activeforeground=FG,
                  relief=tk.FLAT, bd=0, cursor="hand2", pady=7,
                  command=self._save_all).pack(fill=tk.X, padx=10, pady=2)

        tk.Button(sb, text="🗑  Delete  [Delete]",
                  font=_font(10), fg=DANGER, bg=BG3,
                  activebackground=BG3, activeforeground=DANGER,
                  relief=tk.FLAT, bd=0, cursor="hand2", pady=7,
                  command=self._delete_selected).pack(fill=tk.X, padx=10, pady=(2, 8))

        tk.Frame(sb, bg=BG3, height=1).pack(fill=tk.X, padx=6)
        tk.Button(sb, text="Export JSON",
                  font=_font(9), fg=FG2, bg=BG2,
                  activebackground=BG2, activeforeground=FG,
                  relief=tk.FLAT, bd=0, cursor="hand2", pady=4,
                  command=self._export_json).pack(fill=tk.X, padx=10, pady=(6, 4))

        # Statusbar
        self.status_var = tk.StringVar(value="Ready")
        self.coord_var  = tk.StringVar()
        sbar = tk.Frame(self.root, bg=BG3)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(sbar, textvariable=self.status_var, font=_font(9), fg=FG2, bg=BG3,
                 anchor=tk.W).pack(side=tk.LEFT, padx=8, pady=3)
        tk.Label(sbar, textvariable=self.coord_var,  font=_font(9), fg=FG2, bg=BG3,
                 anchor=tk.E).pack(side=tk.RIGHT, padx=8, pady=3)

    def _mk_btn(self, parent, text, cmd, accent=False):
        return tk.Button(parent, text=text, font=_font(9),
                         fg="#fff" if accent else FG,
                         bg=ACCENT if accent else BG3,
                         activebackground=ACCENT, activeforeground="#fff",
                         relief=tk.FLAT, bd=0, cursor="hand2",
                         padx=8, pady=3, command=cmd)

    def _bind_keys(self):
        self.root.bind("<b>",        lambda e: self._set_tool("box"))
        self.root.bind("<v>",        lambda e: self._set_tool("move"))
        self.root.bind("<Control-z>",lambda e: self._undo())
        self.root.bind("<Control-s>",lambda e: self._save_selected())
        self.root.bind("<Control-S>",lambda e: self._save_all())
        self.root.bind("<Delete>",   lambda e: self._delete_selected())
        self.root.bind("<Escape>",   lambda e: self._deselect())

    # ── Tool / color ──────────────────────────────────────────────────────────

    def _set_tool(self, tool):
        self.tool = tool
        self.canvas.config(cursor="crosshair" if tool == "box" else "fleur")
        self.btn_box.config( bg=ACCENT if tool == "box"  else BG3, fg="#fff" if tool=="box"  else FG)
        self.btn_move.config(bg=ACCENT if tool == "move" else BG3, fg="#fff" if tool=="move" else FG)

    def _set_color(self, color):
        self.color = color
        self._highlight_color(color)
        if self.selected:
            self.selected.color = color
            self._redraw(self.selected)

    def _highlight_color(self, active):
        for c, b in self.color_btns:
            b.config(highlightbackground=FG if c == active else BG2,
                     highlightthickness=2 if c == active else 1)

    # ── Canvas events ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        x, y = event.x, event.y
        if self.tool == "move":
            hit = next((a for a in reversed(self.annotations) if a.contains(x, y)), None)
            if hit:
                self._select(hit)
                self.drag_data = {"ann": hit, "ox": x - hit.x1, "oy": y - hit.y1,
                                  "w": hit.w, "h": hit.h}
            else:
                self._deselect()
        else:
            self.drawing   = True
            self.start_xy  = (x, y)

    def _on_drag(self, event):
        x, y = event.x, event.y
        if self.tool == "move" and self.drag_data:
            ann = self.drag_data["ann"]
            nx1 = max(0, min(CANVAS_W - self.drag_data["w"], x - self.drag_data["ox"]))
            ny1 = max(0, min(CANVAS_H - self.drag_data["h"], y - self.drag_data["oy"]))
            ann.x1, ann.y1 = nx1, ny1
            ann.x2, ann.y2 = nx1 + self.drag_data["w"], ny1 + self.drag_data["h"]
            self._redraw(ann)
        elif self.drawing:
            if self.live_rect:
                self.canvas.delete(self.live_rect)
            sx, sy = self.start_xy
            self.live_rect = self.canvas.create_rectangle(
                sx, sy, x, y, outline=self.color, width=2, dash=(5, 3))

    def _on_release(self, event):
        self.drag_data = {}
        if not self.drawing:
            return
        self.drawing = False
        if self.live_rect:
            self.canvas.delete(self.live_rect)
            self.live_rect = None

        sx, sy = self.start_xy
        x1, y1 = min(sx, event.x), min(sy, event.y)
        x2, y2 = max(sx, event.x), max(sy, event.y)

        if x2 - x1 < 5 or y2 - y1 < 5:
            self._set_status("Box too small — try again", warn=True)
            return

        ann = Annotation(x1, y1, x2, y2, color=self.color)
        self.annotations.append(ann)
        self._draw(ann)
        self._select(ann)
        self._refresh_list()
        self.label_entry.focus_set()
        self._set_status(f"Box drawn ({ann.w}×{ann.h}px) — enter a label and save")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, ann):
        lw = 3 if ann is self.selected else 2
        ann.rect_id = self.canvas.create_rectangle(
            ann.x1, ann.y1, ann.x2, ann.y2,
            outline=ann.color, width=lw, tags="ann")
        self._draw_badge(ann)

    def _draw_badge(self, ann):
        if ann.text_id:
            self.canvas.delete(ann.text_id)
            ann.text_id = None
        if ann.label:
            ann.text_id = self.canvas.create_text(
                ann.x1 + 3, ann.y1 - 1,
                text=ann.label, anchor=tk.SW,
                font=_font(8, bold=True), fill=ann.color, tags="ann")

    def _redraw(self, ann):
        if ann.rect_id:
            self.canvas.coords(ann.rect_id, ann.x1, ann.y1, ann.x2, ann.y2)
            self.canvas.itemconfig(ann.rect_id, outline=ann.color,
                                   width=3 if ann is self.selected else 2)
        if ann.text_id:
            self.canvas.delete(ann.text_id)
            ann.text_id = None
        self._draw_badge(ann)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select(self, ann):
        prev = self.selected
        self.selected = ann
        if prev and prev is not ann:
            self._redraw(prev)
        self._redraw(ann)
        self.label_var.set(ann.label)
        idx = self.annotations.index(ann)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)

    def _deselect(self):
        prev, self.selected = self.selected, None
        if prev:
            self._redraw(prev)
        self.listbox.selection_clear(0, tk.END)

    def _on_list_select(self, event):
        idxs = self.listbox.curselection()
        if idxs:
            self._select(self.annotations[idxs[0]])
            self.label_entry.focus_set()

    # ── Label live update ─────────────────────────────────────────────────────

    def _update_label_live(self):
        if self.selected:
            self.selected.label = self.label_var.get()
            self._redraw(self.selected)
            self._refresh_list(keep=True)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _undo(self):
        if not self.annotations:
            return
        ann = self.annotations.pop()
        if ann.rect_id: self.canvas.delete(ann.rect_id)
        if ann.text_id: self.canvas.delete(ann.text_id)
        if self.selected is ann:
            self.selected = None
        self._refresh_list()
        self._set_status("Undone")

    def _delete_selected(self):
        if not self.selected:
            return
        ann = self.selected
        self.annotations.remove(ann)
        if ann.rect_id: self.canvas.delete(ann.rect_id)
        if ann.text_id: self.canvas.delete(ann.text_id)
        self.selected = None
        self._refresh_list()
        self._set_status("Deleted")

    def _clear_all(self):
        if not self.annotations:
            return
        if not messagebox.askyesno("Clear all", "Delete all annotations?"):
            return
        self.canvas.delete("ann")
        self.annotations.clear()
        self.selected = None
        self._refresh_list()
        self._set_status("Cleared")

    def _save_selected(self):
        if not self.selected:
            self._set_status("No annotation selected", warn=True)
            return
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("No label", "Enter a label before saving.")
            self.label_entry.focus_set()
            return
        self.selected.label = label
        self._do_save(self.selected)
        self._refresh_list(keep=True)
        self._redraw(self.selected)

    def _save_all(self):
        if not self.annotations:
            self._set_status("No annotations", warn=True)
            return
        unlabeled = [a for a in self.annotations if not a.label.strip()]
        if unlabeled:
            messagebox.showwarning("Unlabeled", f"{len(unlabeled)} box(es) have no label.")
            return
        for ann in self.annotations:
            self._do_save(ann)
        self._set_status(f"Saved {len(self.annotations)} crops → '{OUTPUT_DIR}/'")

    def _do_save(self, ann):
        region = self.image.crop((int(ann.x1), int(ann.y1), int(ann.x2), int(ann.y2)))
        safe   = "".join(c if c.isalnum() or c in "-_ " else "_" for c in ann.label).strip()
        path   = os.path.join(OUTPUT_DIR, f"{safe}_{ann.id:03d}.png")
        region.save(path)
        print(f"[saved] {path}")
        self._set_status(f"Saved → {path}")
        self.canvas.itemconfig(ann.rect_id, outline=SUCCESS)
        self.root.after(500, lambda: self.canvas.itemconfig(ann.rect_id, outline=ann.color) if ann.rect_id else None)

    def _export_json(self):
        if not self.annotations:
            self._set_status("Nothing to export", warn=True)
            return
        out = {}
        for ann in self.annotations:
            key = f"{ann.label or 'ann'}_{ann.id}"
            out[key] = {
                "label": ann.label,
                "x1": round(ann.x1 / CANVAS_W, 6), "y1": round(ann.y1 / CANVAS_H, 6),
                "x2": round(ann.x2 / CANVAS_W, 6), "y2": round(ann.y2 / CANVAS_H, 6),
                "px": {"x1": int(ann.x1), "y1": int(ann.y1),
                       "x2": int(ann.x2), "y2": int(ann.y2)},
            }
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")],
                                            initialfile="annotations.json")
        if path:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            self._set_status(f"JSON exported → {path}")

    # ── Listbox ───────────────────────────────────────────────────────────────

    def _refresh_list(self, keep=False):
        sel = self.listbox.curselection()
        self.listbox.delete(0, tk.END)
        for ann in self.annotations:
            self.listbox.insert(tk.END, f"  {ann.label or '(unlabeled)'}  [{ann.w}×{ann.h}]")
            self.listbox.itemconfig(tk.END, fg=ann.color)
        if keep and sel:
            try:
                self.listbox.selection_set(sel[0])
            except Exception:
                pass

    # ── New screenshot ────────────────────────────────────────────────────────

    def _new_screenshot(self):
        self._set_status("Taking screenshot in 2s — switch window now…")
        self.root.iconify()
        self.root.after(2100, self._finish_screenshot)

    def _finish_screenshot(self):
        try:
            screen      = pyautogui.screenshot()
            self.image  = crop_and_resize(screen)
            self.tk_img = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
            self.annotations.clear()
            self.selected = None
            self._refresh_list()
            self._set_status("New screenshot loaded")
        except Exception as e:
            self._set_status(f"Screenshot failed: {e}", warn=True)
        finally:
            self.root.deiconify()

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg, warn=False):
        self.status_var.set(("⚠  " if warn else "") + msg)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Taking screenshot in 2 seconds — switch to the target window now…")
    screen  = take_screenshot()
    cropped = crop_and_resize(screen)

    root = tk.Tk()
    app  = AnnotatorApp(root, cropped)
    root.mainloop()
    print(f"\nDone. Check '{OUTPUT_DIR}/' for saved crops.")

if __name__ == "__main__":
    main()