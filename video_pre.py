import ffmpeg
import os
from argparse import ArgumentParser, BooleanOptionalAction
'''
Extract a frame on a point of each input video for our COLMAP reconstruction
For the selected camera, extract frames starting from that point.
'''

def main(args):

    frames_folder = os.path.join(args.output, "frames")
    colmap_folder = os.path.join(args.output, "colmap")

    # Create dirs
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(colmap_folder, exist_ok=True)
    os.makedirs(os.path.join(colmap_folder, "input"), exist_ok=True)
    colmap_folder = os.path.join(colmap_folder, "input")
    
    # Camera selection
    if args.camera is None:
        camera = os.listdir(args.source)[0].split(".")[0]
        print(f"No camera selected, using {camera} as default")
    else: 
        camera = args.camera
    
    # Extract frames: selected cam -> all, others -> 1 frame
    for video in os.listdir(args.source):
        
        if not video.endswith(".mp4"):
            print(f"Skipping {video} as it is not an mp4 file")
            continue
        
        name = video.split(".")[0]

        if name == camera:
            stream = ffmpeg.input(os.path.join(args.source, video), ss=args.time)
            os.makedirs(os.path.join(frames_folder, name), exist_ok=True)
            out_path = os.path.join(frames_folder, name, "%04d.png")
            print(f"Extracting frames from {video}")
            stream = stream.filter("scale", args.resize_width, -1)
            stream = ffmpeg.output(stream, out_path, loglevel="error")
        
            ffmpeg.run(stream)
        
        stream = ffmpeg.input(os.path.join(args.source, video), ss=args.time, t=1).filter("fps", fps=1)
        out_path = os.path.join(colmap_folder, f"{name}.png")
        print(f"Extracting single frame from {video}")
        stream = stream.filter("scale", args.resize_width, -1)
        stream = ffmpeg.output(stream, out_path, loglevel="error")
    
        ffmpeg.run(stream)

    print("Done!")

if __name__== "__main__":
    ap = ArgumentParser()
    ap.add_argument("-s", "--source",type=str, help="path to the input videos folder", default="dataset")
    ap.add_argument("-t", "--time", type=int, help="starting time to extract (seconds)", default=0)
    ap.add_argument("-c", "--camera", type=str, help="camera to extract all frames from")
    ap.add_argument("-o", "--output", type=str, help="path to the output directory", default="dataset_processed")
    ap.add_argument("-r", "--resize-width", type=int, help="resize width", default=1400)
    args = ap.parse_args()

    main(args)