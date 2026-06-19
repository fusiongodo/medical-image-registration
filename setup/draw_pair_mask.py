import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from setup.pair_mask import (
    downsample_page,
    image_to_page_coords,
    load_he_page_gray,
    load_pair_target,
    mask_json_path,
    mask_preview_path,
    rasterize_polygon,
    save_mask_preview,
    save_pair_mask,
)

PAIR_ID = 19
PYRAMID_PAGE_IDX = 4
DISPLAY_MAX_SIDE = 8192
FIGSIZE = (24, 18)
CLOSE_PIXELS = 24
ZOOM_BASE = 1.15


class PolygonDrawer:
    def __init__(self, ax, on_complete, close_pixels=CLOSE_PIXELS):
        self.ax = ax
        self.on_complete = on_complete
        self.close_pixels = close_pixels
        self.points = []
        self.completed = False
        self.panning = False
        self.pan_anchor = None
        self.pan_xlim = None
        self.pan_ylim = None

        self.vertex_artists = []
        self.edge_line, = ax.plot(
            [], [], color="lime", linewidth=2.5, alpha=0.95, zorder=12,
        )
        self.fill_patch = None
        self.first_marker, = ax.plot(
            [], [], marker="o", markersize=14, markerfacecolor="none",
            markeredgecolor="yellow", markeredgewidth=3, linestyle="none", zorder=13,
        )

        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self.on_press),
            canvas.mpl_connect("button_release_event", self.on_release),
            canvas.mpl_connect("motion_notify_event", self.on_motion),
            canvas.mpl_connect("scroll_event", self.on_scroll),
            canvas.mpl_connect("key_press_event", self.on_key),
        ]

    def pixel_distance(self, x0, y0, x1, y1):
        p0 = self.ax.transData.transform((x0, y0))
        p1 = self.ax.transData.transform((x1, y1))
        return float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))

    def refresh(self):
        for artist in self.vertex_artists:
            artist.remove()
        self.vertex_artists.clear()

        if self.fill_patch is not None:
            self.fill_patch.remove()
            self.fill_patch = None

        if not self.points:
            self.edge_line.set_data([], [])
            self.first_marker.set_data([], [])
            self.ax.figure.canvas.draw_idle()
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self.edge_line.set_data(xs, ys)
        self.first_marker.set_data([xs[0]], [ys[0]])

        for x, y in self.points:
            marker, = self.ax.plot(
                x, y, marker="o", markersize=10,
                markerfacecolor="lime", markeredgecolor="yellow",
                markeredgewidth=2, linestyle="none", zorder=14,
            )
            self.vertex_artists.append(marker)

        if self.completed and len(self.points) >= 3:
            self.fill_patch = MplPolygon(
                self.points, closed=True,
                facecolor="lime", edgecolor="lime",
                alpha=0.25, linewidth=2, zorder=11,
            )
            self.ax.add_patch(self.fill_patch)

        self.ax.figure.canvas.draw_idle()

    def complete(self):
        if len(self.points) < 3 or self.completed:
            return
        self.completed = True
        self.refresh()
        self.on_complete(list(self.points))

    def on_press(self, event):
        if event.inaxes != self.ax or self.completed:
            return

        if event.button in (2, 3):
            self.panning = True
            self.pan_anchor = (event.x, event.y)
            self.pan_xlim = self.ax.get_xlim()
            self.pan_ylim = self.ax.get_ylim()
            return

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        if len(self.points) >= 3:
            x0, y0 = self.points[0]
            if self.pixel_distance(x0, y0, x, y) <= self.close_pixels:
                self.complete()
                return

        self.points.append((x, y))
        self.refresh()

    def on_release(self, event):
        self.panning = False
        self.pan_anchor = None

    def on_motion(self, event):
        if not self.panning or self.pan_anchor is None:
            return
        if event.x is None or event.y is None:
            return

        dx = event.x - self.pan_anchor[0]
        dy = event.y - self.pan_anchor[1]
        x0, x1 = self.pan_xlim
        y0, y1 = self.pan_ylim
        inv = self.ax.transData.inverted()
        p0 = inv.transform((self.pan_anchor[0], self.pan_anchor[1]))
        p1 = inv.transform((event.x, event.y))
        shift_x = p1[0] - p0[0]
        shift_y = p1[1] - p0[1]
        self.ax.set_xlim(x0 - shift_x, x1 - shift_x)
        self.ax.set_ylim(y0 - shift_y, y1 - shift_y)
        self.ax.figure.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale = 1 / ZOOM_BASE if event.button == "up" else ZOOM_BASE
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        new_width = width * scale
        new_height = height * scale

        relx = (xlim[1] - event.xdata) / width
        rely = (ylim[1] - event.ydata) / height
        self.ax.set_xlim(
            event.xdata - new_width * (1 - relx),
            event.xdata + new_width * relx,
        )
        self.ax.set_ylim(
            event.ydata - new_height * (1 - rely),
            event.ydata + new_height * rely,
        )
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        if self.completed:
            return
        if event.key == "u" and self.points:
            self.points.pop()
            self.refresh()
        elif event.key == "c":
            self.points.clear()
            self.refresh()
        elif event.key in ("enter", "return") and len(self.points) >= 3:
            self.complete()


def main():
    target_id, path = load_pair_target(PAIR_ID)
    page = load_he_page_gray(path, PYRAMID_PAGE_IDX)
    page_h, page_w = page.shape[:2]
    display, scale = downsample_page(page, DISPLAY_MAX_SIDE)
    disp_h, disp_w = display.shape[:2]

    saved = {"polygon": None}

    def on_complete(verts):
        saved["polygon"] = image_to_page_coords(verts, scale)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.canvas.manager.set_window_title(f"Pair {PAIR_ID} HE mask")
    ax.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title(
        f"pair {PAIR_ID} | HE {target_id} | page {PYRAMID_PAGE_IDX} | "
        f"display {disp_w}x{disp_h} (page {page_w}x{page_h})"
    )
    ax.set_xlim(-0.5, disp_w - 0.5)
    ax.set_ylim(disp_h - 0.5, -0.5)

    help_text = (
        "Left-click: add vertex (lime dots)\n"
        "Click first point again: close polygon\n"
        "Scroll: zoom at cursor\n"
        "Right-drag: pan\n"
        "u: undo  |  c: clear  |  Enter: close"
    )
    ax.text(
        0.01, 0.01, help_text, transform=ax.transAxes,
        fontsize=11, color="white", va="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 6},
        zorder=20,
    )

    PolygonDrawer(ax, on_complete)

    print(f"Pair ID         : {PAIR_ID}")
    print(f"HE target       : {target_id}")
    print(f"Pyramid page    : {PYRAMID_PAGE_IDX}")
    print(f"Page size       : {page_w} x {page_h}")
    print(f"Display size    : {disp_w} x {disp_h} (scale {scale:.4f})")
    print(f"Output          : {mask_json_path(PAIR_ID)}")
    print()
    print("Controls:")
    print("  Left-click       add vertex (lime markers appear immediately)")
    print("  Click first pt   close polygon")
    print("  Scroll wheel     zoom in/out at cursor")
    print("  Right-drag       pan")
    print("  u / c / Enter    undo / clear / close")
    print()
    print("Close the window when the polygon is closed (green fill).")
    print()

    plt.tight_layout()
    plt.show()

    polygon = saved["polygon"]
    if not polygon or len(polygon) < 3:
        print("[ERROR] No polygon saved — close the polygon before closing the window.")
        sys.exit(1)

    out_path = save_pair_mask(
        pair_id=PAIR_ID,
        target_image_id=target_id,
        pyramid_page_idx=PYRAMID_PAGE_IDX,
        page_width=page_w,
        page_height=page_h,
        polygon=polygon,
    )

    mask = rasterize_polygon(polygon, page_w, page_h)
    save_mask_preview(PAIR_ID, mask)

    print(f"Saved polygon : {out_path}")
    print(f"Preview mask  : {mask_preview_path(PAIR_ID)}")


if __name__ == "__main__":
    main()
