import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, default=2, help="Base number (default: 2).")
parser.add_argument("--num", type=int, default=1_000, help="Number of fractions per 1 in an axis (default: 1,000).")
parser.add_argument("--max-iter", type=int, default=50, help="Number of y-axis (default: 50).")
parser.add_argument("--threshold", type=float, default=1e-5, help="Threshold (default: 0.00001).")
parser.add_argument("--escape-radius", type=float, default=1e+10, help="Escape radius (default: 1,000,000,000).")
parser.add_argument("--min-x", type=int, default=4, help="Min of x-axis (default: 4).")
parser.add_argument("--max-x", type=int, default=6, help="Max of x-axis (default: 6).")
parser.add_argument("--min-y", type=int, default=-1, help="Min of y-axis (default: -1).")
parser.add_argument("--max-y", type=int, default=1, help="Max of y-axis (default: 1).")
parser.add_argument("--plot", type=bool, default=True, help="Show plot (default: True).")
parser.add_argument("--cmap", type=str, default="binary", help="cmap (default: 'binary).")
parser.add_argument("--savefig", type=bool, default=True, help="Show plot figure (default: True).")
parser.add_argument("--save-dir", type=str, default="out", help="Save directory (default: 'out').")
parser.add_argument("--plt-imshow-origin", type=str, default="lower", help="Matplot imshow origin (default: 'lower').")
args = parser.parse_args()

# Function to check if a sequence converges
def is_convergent(seq : list, threshold : float) -> bool:
    """Summary about is_convergent

    :param seq: Sequence
    :type seq: list
    :param threshold: Threshold
    :type threshold: float
    :return: np.abs(seq[-1] - seq[-2]) < threshold
    :rtype: bool
    """
    # Check if the absolute difference between last two elements is below the threshold
    return np.abs(seq[-1] - seq[-2]) < threshold

# Function to perform the tetration opeation and check for convergence
def compute_tetration_convergence(base : np.float64, nx : int, ny : int, max_iter : int, threshold : float, escape_radius : float) -> np.zeros_like:
    """Summary about compute_tet...

    :param base: 
    :type base: np.float64
    :param nx: 
    :type nx: int
    :param ny: 
    :type ny: int
    :param max_iter: 
    :type max_iter: int
    :param threshold: 
    :type threshold: float
    :param escape_radius: 
    :type escape_radius: float
    :return: 
    :rtype: np.zeros_like
    """
    # Expeanding the range for real and imaginary parts
    x = np.linspace(args.min_x, args.max_x, nx)
    y = np.linspace(args.min_y, args.max_y, ny)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    convergence_map = np.zeros_like(c, dtype=bool)
    for i in tqdm(range(nx), desc="nx"):
        for j in range(ny):
            z = c[i, j]
            seq = [z]
            for k in range(max_iter):
                z = base ** z
                seq.append(z)
                if np.abs(z) > escape_radius or is_convergent(seq, threshold):
                    break
            convergence_map[i, j] = is_convergent(seq, threshold)
    return convergence_map

# Main
if __name__ == "__main__":
    # Compute the convergence map
    convergence_map = compute_tetration_convergence( # Parameters
        base = np.sqrt(args.base), 
        nx = int((args.max_x - args.min_x) * args.num), 
        ny = int((args.max_y - args.min_y) * args.num), 
        max_iter = args.max_iter, 
        threshold = args.threshold, 
        escape_radius = args.escape_radius)
    # Plotting
    plt.imshow(convergence_map.T, extent=[args.min_x, args.max_x, args.min_y, args.max_y], origin=args.plt_imshow_origin, cmap=args.cmap)
    plt.xlabel("Real Number")
    plt.ylabel("Imaginary Number")
    plt.title(f"Convergence and Divergence in Tetration with √{args.base}")
    plt.colorbar(label="Divergence (0) / Convergence (1)")
    if args.savefig: plt.savefig(fname=f"{args.save_dir}/tetration_fractal({time.strftime('%Y-%m-%d %H-%M-%S')}).png")
    if args.plot: plt.show()