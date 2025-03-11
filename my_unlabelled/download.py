import yt_dlp

video_url = "https://www.youtube.com/watch?v=L4Mam-uKRYE"

ydl_opts = {
    'format': 'bestvideo[height<=720]',  # Change 2160 to your desired max resolution
    'outtmpl': './vids/video.mp4',
    'merge_output_format': 'mp4'  # Ensure final output is MP4
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Download complete! Video saved as video.mp4")