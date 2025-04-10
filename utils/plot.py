import warnings

import torch


from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import ImageColor


def image(img: torch.Tensor, title: str = None):
    plt.imshow(img.permute(1, 2, 0) / 255)
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])


def image_pair(img1: torch.Tensor,
                    img2: torch.Tensor):
    img1 = img1 / 255
    img2 = img2 / 255

    fig, axis = plt.subplots(ncols=2)
    axis[0].imshow(img1.permute(1, 2, 0))
    axis[1].imshow(img2.permute(1, 2, 0))


def grid(images: torch.Tensor,
              main_title: str = None,
              titles: list[str] = None,
              cols: int = None, rows: int = None,
              size_multiplier: int = 2,
              pad: tuple[float, float] = None):
    im_cnt = images.shape[0]
    if cols is not None and rows is not None:
        fig_size = (rows, cols)
    elif cols is not None:
        fig_size = (im_cnt // cols + (im_cnt % cols > 0), cols)
    elif rows is not None:
        fig_size = (rows, im_cnt // rows + (im_cnt % rows > 0))
    else:
        cols = min(4, im_cnt)
        fig_size = (im_cnt // cols + (im_cnt % cols > 0), cols)

    fig = plt.figure(figsize=(fig_size[1] * size_multiplier, 
                              fig_size[0] * size_multiplier), 
                              constrained_layout=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=fig_size,  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    if main_title is not None:
        fig.suptitle(main_title)

    for idx, (ax, im) in enumerate(zip(grid, images)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im.permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.tight_layout()

    if pad is not None:
        grid.set_axes_pad(pad)


def hex_to_cv2(hex_col: str):
    """
    Convert HEX color to BGR

    Parameters
    ----------
    hex_col : str
        HEX string
    """
    return ImageColor.getcolor(hex_col, "RGB")[::-1]
