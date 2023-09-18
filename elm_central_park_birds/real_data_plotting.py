import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.collections import LineCollection

def generate_trajectory_video(dataframe, output_filename='bird_trajectories.mp4', fps=30, n_frames=None):
    sorted_data = dataframe.sort_values(by='time')
    unique_birds = sorted_data['bird'].unique()
    colors = sns.color_palette("dark", len(unique_birds))
    
    # Create initial plot frame
    fig, ax = plt.subplots()
    fig.set_facecolor('darkgray')  # Dark gray figure background
    
    line_collections = []
    for idx, bird_id in enumerate(unique_birds):
        bird_data = sorted_data[sorted_data['bird'] == bird_id]
        lc = LineCollection([], colors=colors[idx], linewidths=3, alpha=0.8)
        line_collections.append(lc)
        ax.add_collection(lc)

    def update(num):
        for idx, bird_id in enumerate(unique_birds):
            bird_data = sorted_data[sorted_data['bird'] == bird_id]
            if n_frames:
                bird_data = bird_data.iloc[:min(n_frames, num+1)]
            else:
                bird_data = bird_data.iloc[:num+1]
            
            x_vals = bird_data['x'].values
            y_vals = bird_data['y'].values
            segments = [((x_vals[i-1], y_vals[i-1]), (x_vals[i], y_vals[i])) for i in range(1, len(x_vals))]
            line_collections[idx].set_segments(segments)
            line_collections[idx].set_linewidth([3*(i/len(x_vals)) for i in range(len(segments))])

    ax.set_xlim([0, 350])
    ax.set_ylim([250, 0])  # Flipped y-axis
    ax.axis('off')  # Remove the axes
    ax.set_facecolor('darkgray')  # Dark gray background
    ax.set_aspect('equal', 'box')  # Consistent aspect ratio

    ani_frames = len(sorted_data['time'].unique()) if n_frames is None else n_frames
    ani = animation.FuncAnimation(fig, update, frames=ani_frames, blit=False, interval=1000/fps, repeat=False)
    
    # Save to .mp4
    ani.save(output_filename, writer="ffmpeg", fps=fps)

    plt.close(fig)  # Close the figure after saving

# Example usage:
# generate_trajectory_video(df, 'bird_trajectories.mp4', fps=30)



# def generate_trajectory_video(dataframe, output_filename='bird_trajectories.mp4', fps=30, n_frames=None):
#     def update(num, data, ax):
#         ax.clear()
#         unique_birds = data['bird'].unique()
#         colors = sns.color_palette("dark", len(unique_birds))
        
#         for idx, bird_id in enumerate(unique_birds):
#             bird_data = data[data['bird'] == bird_id]
            
#             if n_frames:
#                 bird_data = bird_data.iloc[:min(n_frames, num+1)]
#             else:
#                 bird_data = bird_data.iloc[:num+1]
            
#             x_vals = bird_data['x'].values
#             y_vals = bird_data['y'].values
            
#             for i in range(1, len(x_vals)):
#                 ax.plot(x_vals[i-1:i+1], y_vals[i-1:i+1], color=colors[idx], alpha=0.8, linewidth=3*(i/len(x_vals)))
        
#         ax.set_xlim([0, 350])
#         ax.set_ylim([250, 0])  # Flipped y-axis
#         ax.axis('off')  # Remove the axes
#         ax.set_facecolor('darkgray')  # Dark gray background
#         ax.set_aspect('equal', 'box')  # Consistent aspect ratio

#     # Create initial plot frame
#     fig, ax = plt.subplots()
#     fig.set_facecolor('darkgray')  # Dark gray figure background
    
#     sorted_data = dataframe.sort_values(by='time')

#     ani_frames = len(sorted_data['time'].unique()) if n_frames is None else n_frames

#     ani = animation.FuncAnimation(fig, update, frames=ani_frames, 
#                                   fargs=[sorted_data, ax], blit=False, interval=1000/fps, repeat=False)
    
#     # Save to .mp4
#     ani.save(output_filename, writer="ffmpeg", fps=fps)

#     plt.close(fig)  # Close the figure after saving