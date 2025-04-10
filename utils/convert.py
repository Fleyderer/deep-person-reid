"""
Helpful conversion functions
"""
import torch
import torchvision.transforms as T
import numpy as np


def tensor_to_np(im_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert Tensor to numpy object depending on whether
    gradient calculation is required


    Parameters
    ----------
    im_tensor : torch.Tensor
        Image tensor


    Returns
    -------
    np.ndarray
        Image array
    """
    if im_tensor.requires_grad:
        return im_tensor.detach().cpu().numpy()
    else:
        return im_tensor.cpu().numpy()


def tensor_to_cv2(im_tensor: torch.Tensor,
                  conversion: str = "RGB") -> np.ndarray:
    """
    Function for conversion tensor to cv2 readable image with 3 channels


    Parameters
    ----------
    im_tensor : torch.Tensor
        Image tensor
    conversion : str
        Conversion parameter. If 'RGB' then firstly convert to RGB
        and secondly convert to BGR and return in cv2 format


    Returns
    -------
    np.ndarray
        Image array
    """
    # Convert the tensor to a PIL image (it helps with RGB handling)
    transform = T.ToPILImage()
    res_image = transform(im_tensor).convert(conversion)

    # Convert to cv2: RGB -> BGR
    if conversion == 'RGB':
        res_image = np.array(res_image)[:, :, ::-1]
    else:
        res_image = np.array(res_image)

    return res_image


def cv2_to_tensor(im_cv2: np.ndarray,
                  axis=['H', 'W', 'C'],
                  max_c=3) -> torch.Tensor:
    """
    Convert numpy image to tensor


    Parameters
    ----------
    im_cv2 : np.ndarray
        Numpy image
    axis : list, optional
        Types of input axis, by default ['H', 'W', 'C']
        Usually cv2 has format [H, W, C], but tensors should have [C, H, W],
        so inside function we convert it to [C, H, W]
    max_c : int, optional
        Max count of channels to save, by default 3


    Returns
    -------
    torch.Tensor
        Tensor with image, having [C, H, W] format
    """
    if len(im_cv2.shape) == 3:

        # BGR to RGB
        im = im_cv2[:, :, ::-1]

        return torch.from_numpy(im.copy()).permute(axis.index('C'),
                                                   axis.index('H'),
                                                   axis.index('W'))[:max_c]

    elif len(im_cv2.shape) == 4:
        if len(axis) != 4:
            axis = ['B'] + axis
        im = im_cv2[:, :, :, ::-1]

        return torch.from_numpy(im.copy()).permute(axis.index('B'),
                                                   axis.index('C'),
                                                   axis.index('H'),
                                                   axis.index('W'))[:, :max_c]


# Following functions are from
# https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/21
def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """
    N that maps from normalized to unnormalized coordinates
    """
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)


def m_to_theta(M, w, h):
    """
    convert affine warp matrix `M` compatible with `opencv.warpAffine`
    to `theta` matrix compatible with `torch.F.affine_grid`

    
    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

        
    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def theta_to_m(theta, w, h, return_inv=False):
    """
    convert theta matrix compatible with `torch.F.affine_grid` 
    to affine warp matrix `M` compatible with `opencv.warpAffine`.

    
    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using 
    `opencv.perspectiveTransform`, M^-1 is required

    
    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

        
    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]