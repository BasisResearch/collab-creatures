import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def animate_trajectories(sim, frames_skipped = 1, fMin=None, fMax = None):

    x = sim.trajectories[:, :, 0]  # shape : nfish x time
    y = sim.trajectories[:, :, 1]
    theta = sim.velocities[:, :, 1]
    N = sim.N
    arena_size = sim.arena_size
    dt = sim.dt
    L = sim.arena_size/10

    if fMin is None:
        fMin = 0
    if fMax is None:
        fMax = x.shape[1]


    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-arena_size, arena_size)
    ax.set_ylim(-arena_size, arena_size)
    ax.set_aspect("equal")

    # Initialize scatter for particle positions and lines for velocity directions
    particles = ax.scatter(x[:, 0], y[:, 0], color="blue")
    velocity_lines = [ax.plot([], [], color="blue")[0] for _ in range(N)]

    # plot arena outline
    temp = np.linspace(0, 2 * np.pi, 100)
    ax.plot(arena_size * np.cos(temp), arena_size * np.sin(temp), "k")

    # Update function for each frame
    def update(frame):
        # Update the particle positions
        particles.set_offsets(np.c_[x[:, frame], y[:, frame]])

        # Update velocity direction lines
        for i, line in enumerate(velocity_lines):
            dx = -L * np.cos(theta[i, frame])
            dy = -L * np.sin(theta[i, frame])
            line.set_data(
                [x[i, frame], x[i, frame] + dx], [y[i, frame], y[i, frame] + dy]
            )
            ax.set_title(f"t={frame*dt : .0f}")
        return particles, *velocity_lines

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(fMin,fMax,frames_skipped), interval=1/dt, blit=True)

    # Display the animation inline in Jupyter Notebook
    return HTML(anim.to_jshtml())
