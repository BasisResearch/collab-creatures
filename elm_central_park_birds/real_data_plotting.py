import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_bird_trajectories_video(data, output_path, fps=30, frames=None, max_history=100):
    # Sorting data
    sorted_data = data.sort_values(by="time")
    
    # Getting unique bird IDs
    unique_birds = sorted_data['bird'].unique()

    # Colors for birds
    colors = sns.color_palette("dark", len(unique_birds))

    # Determine frames to consider
    if frames is None:
        frames = len(sorted_data['time'].unique())
    else:
        frames = min(frames, len(sorted_data['time'].unique()))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    # Warm background color
    background_color = '#FAEBD7'
    fig.patch.set_facecolor(background_color)

    # Set limits and aspect ratio
    ax.set_xlim(0, 350)
    ax.set_ylim(250, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    def animate(frame):
        # Start frame
        start_frame = max(0, frame - max_history)
        
        current_data = sorted_data[(sorted_data['time'] <= frame) & (sorted_data['time'] >= start_frame)]
        
        
        # Clear all scatter plots and lines
        for coll in ax.collections:
            coll.remove()
            
        for line in ax.lines:
            line.remove()
            
        for idx, bird_id in enumerate(unique_birds):
            bird_data = current_data[current_data['bird'] == bird_id]
            if len(bird_data) == 0:
                continue

            # Compute sizes based on how old the point is
            sizes = np.linspace(1, 10, len(bird_data)) ** 2.5
            # Create new scatter for this bird
            ax.scatter(bird_data['x'], bird_data['y'], color=colors[idx], s=sizes, alpha=0.25, edgecolor='none')
            # Plot lines
            ax.plot(bird_data['x'], bird_data['y'], color=colors[idx], alpha=0.25, linewidth=1)




                
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, repeat=False)

    ani.save(output_path, writer=writer, dpi=300)

    plt.close(fig)
