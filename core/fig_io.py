from pathlib import Path

import matplotlib.pyplot as plt


def show_or_save_figure(fig, save_path: Path | None = None, dpi: int = 150) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    else:
        plt.show()
    plt.close(fig)
