from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import tdgl
from tdgl.visualization.animate import create_animation


def make_video_from_solution(
    solution,
    name,
    # quantities=("normal_current", "scalar_potential"),
    quantities=("order_parameter", "scalar_potential"),
    fps=20,
    figsize=(5, 4),
):
    """Generates an HTML5 video from a tdgl.Solution."""
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            video = anim.save(name, fps=fps, writer="ffmpeg")
  
