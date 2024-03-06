import numpy as np
import imageio
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True, help="Path to the sample video")
    parser.add_argument("--target", type=str, required=False, help="Path to the target video")
    parser.add_argument("--thr1", type=float, default=0.1, help="Canny (korina impl.) edge detector threshold1")
    parser.add_argument("--thr2", type=float, default=0.2, help="Canny (korina impl.) edge detector threshold2")
    parser.add_argument("--sc", type=bool, default=True, help="Usage of shift compensation")
    parser.add_argument("--pwe", type=bool, default=False, help="Penalize wider edges")
    parser.add_argument("--gc", type=bool, default=False, help="Global compensation")
    parser.add_argument("--isYUV", type=bool, default=False, help="Set true if the video is in YUV format")
    parser.add_argument("--resolution", type=str, help="Set the resolution if the video is in YUV format")
    parser.add_argument("--filepath", type=str, default=None, help="Path to file of result")
    parser.add_argument("--tmp", type=str, default=None, help="Path to tmp files directory")
    return parser.parse_args()

args = parse_args()
assert os.path.exists(args.sample), f"No such video {args.sample}"
assert os.path.exists(args.target), f"No such image {args.target}"
inp = None
if not args.isYUV:
    vid1 = imageio.get_reader(args.target, 'ffmpeg')
else:
    inp = os.path.join(args.tmp, 'input.avi')
    os.system(f"ffmpeg -y -s {args.resolution} -pixel_format yuv420p -i {args.target} -vcodec rawvideo -pix_fmt bgr24 {inp}")
    vid1 = imageio.get_reader(inp, 'ffmpeg')
vid2 = imageio.get_reader(args.sample, 'ffmpeg')
cnt = 0
for frame in vid1:
    cnt += 1
num = 0
os.chdir("pieapp")
score1 = 0.
score2 = 0.
for framegt in vid1:
    framedist = vid2.get_data(num)
    num += 1
    print(f"Processing {num}/{cnt}")
    imageio.imwrite("gt.png", framegt)
    imageio.imwrite("dist.png", framedist)
    os.system("python test_PieAPP_TF.py --ref_path gt.png --A_path dist.png")
    with open("res.txt", "r") as f:
        score1 += float(f.readline())
    os.system("python test_PieAPP_PT.py --ref_path gt.png --A_path dist.png")
    with open("res.txt", "r") as f:
        score2 += float(f.readline())
score1 /= cnt; score2 /= cnt
os.chdir(os.pardir)
if args.filepath:
    with open(args.filepath, 'w') as f:
        f.write(str(score1) + ' ' + str(score2))
else:
    print(score1, '\n', score2)  
