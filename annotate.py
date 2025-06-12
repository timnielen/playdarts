import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np

colors = ["purple", "red", "orange", "yellow", "green"]
markersize = 20
def annotate_images(directory):
    output_path = os.path.join(directory, "labels.pkl")

    if os.path.exists(output_path):
        df = pd.read_pickle(output_path)
    else:
        df = pd.DataFrame(columns=["filename", "locations", "num_darts"])

    image_paths = sorted(glob.glob(os.path.join(directory, "*.[pjJP][pnNP]*[gG]")))

    coords = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)

        image = Image.open(image_path)
        width, height = image.size

        fig, ax = plt.subplots()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        ax.imshow(image)
        
        # Load existing annotations if present
        existing = df[df['filename'] == filename]
        print(df)
        if not existing.empty:
            print(f"Editing existing annotations for {filename}")
            # print(coords)
            coords = existing.iloc[0]['locations']
            print(coords, len(coords))
        else:
            print(f"Annotating new file: {filename}")
        
        markers = []
        dragging = {
            "index": None,
            "marker": None
        }
        for i, (x_rel, y_rel) in enumerate(coords):
            color_idx = min(i, len(colors) - 1)
            x_img = x_rel * width
            y_img = (1 - y_rel) * height
            marker, = ax.plot(x_img, y_img, marker='x', color=colors[color_idx], markersize=markersize)
            markers.append(marker)


        print(f"Left-click to add, right-click to remove, Ctrl+Z to undo last. Press Enter to save.")
        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return

            x_img, y_img = event.xdata, event.ydata
            x_rel = x_img / width
            y_rel = 1 - y_img / height

            if event.button == 1:  # Left-click
                # Check if clicking near existing point for dragging
                for i, (x, y) in enumerate(coords):
                    x_px = x * width
                    y_px = (1 - y) * height
                    if np.hypot(x_px - x_img, y_px - y_img) < 10:
                        dragging["index"] = i
                        dragging["marker"] = markers[i]
                        return

                # Else: Add new point
                coords.append((x_rel, y_rel))
                insert_idx = len(coords) - 1
                color_idx = min(insert_idx, len(colors) - 1)
                marker, = ax.plot(x_img, y_img, marker='x', color=colors[color_idx], markersize=markersize)
                markers.append(marker)
                fig.canvas.draw()

            elif event.button == 3:  # Right-click to remove nearest
                if not coords:
                    return
                distances = [
                    np.hypot((x * width) - x_img, (1 - y) * height - y_img)
                    for x, y in coords
                ]
                min_idx = np.argmin(distances)
                if distances[min_idx] < 10:
                    coords.pop(min_idx)
                    for marker in markers:
                        marker.remove()
                    markers.clear()
                    for i, (x_rel, y_rel) in enumerate(coords):
                        color_idx = min(i, len(colors) - 1)
                        x_img = x_rel * width
                        y_img = (1 - y_rel) * height
                        marker, = ax.plot(x_img, y_img, marker='x', color=colors[color_idx], markersize=markersize)
                        markers.append(marker)
                    fig.canvas.draw()
                    print(f"Removed point #{min_idx + 1}")

        def on_key(event):
            if event.key == 'enter':
                plt.close()
            elif event.key == 'escape':
                plt.close('all')
                print("Annotation cancelled.")
                exit(0)
            elif event.key == 'ctrl+z' or (event.key == 'z' and event.ctrl):
                if coords:
                    coords.pop()
                    for marker in markers:
                        marker.remove()
                    markers.clear()
                    for i, (x_rel, y_rel) in enumerate(coords):
                        color_idx = min(i, len(colors) - 1)
                        x_img = x_rel * width
                        y_img = (1 - y_rel) * height
                        marker, = ax.plot(x_img, y_img, marker='x', color=colors[color_idx], markersize=markersize)
                        markers.append(marker)
                    fig.canvas.draw()
                    print("Removed last annotation.")
                    
        def on_release(event):
            dragging["index"] = None
            dragging["marker"] = None
            
        def on_motion(event):
            if dragging["index"] is not None and event.xdata and event.ydata:
                idx = dragging["index"]
                x_img = event.xdata
                y_img = event.ydata
                x_rel = x_img / width
                y_rel = 1 - y_img / height
                coords[idx] = (x_rel, y_rel)

                dragging["marker"].set_xdata([x_img])
                dragging["marker"].set_ydata([y_img])
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        plt.show()

        # Remove previous entry for this file (if editing)
        df = df[df['filename'] != filename]

        entry = {
            "filename": filename,
            "locations": coords.copy(),
            "num_darts": max(0, len(coords) - 4)
        }

        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        print(df)
        df.to_pickle(output_path)
        print(f"Saved: {filename} with {len(coords)} points.")

    print("Annotation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image annotation tool")
    parser.add_argument("directory", type=str, help="Path to image directory")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
    else:
        annotate_images(args.directory)
