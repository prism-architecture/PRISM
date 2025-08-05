import cv2
import os

def make_video_from_images(folder, output_video, fps=30, target_resolution=(854, 480)):
    print(f'Processing: {folder}')
    
    entries = sorted(
        [entry for entry in os.scandir(folder) if entry.is_file()],
        key=lambda e: e.stat().st_ctime
    )

    if not entries:
        print("No files found.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(folder, output_video)
    out = cv2.VideoWriter(video_path, fourcc, fps, target_resolution)

    for entry in entries:
        frame = cv2.imread(entry.path)
        if frame is None:
            print(f"Skipping unreadable file: {entry.name}")
            continue
        resized_frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    out.release()
    print(f"Video saved at: {video_path}")

def process_all_episodes(root_path):
    task_name = os.path.basename(os.path.normpath(root_path))
    
    for entry in os.listdir(root_path):
        episode_path = os.path.join(root_path, entry)
        recording_path = os.path.join(episode_path, "recording")

        if os.path.isdir(recording_path):
            video_name = f"{task_name}_{entry}.mp4"
            make_video_from_images(recording_path, output_video=video_name)
        else:
            print(f"Skipping (no recording folder): {episode_path}")

# Example usage
process_all_episodes("give-edisode-link-here-from-data-folder")
