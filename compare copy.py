import argparse
import os
import json
parser = argparse.ArgumentParser()
parser.add_argument('--base_path')
parser.add_argument('--ours_path')
parser.add_argument('--threshold', '-t', type=float)
args = parser.parse_args()
with open(os.path.join(args.base_path, "per_view.json"), 'r') as f:
    base_data = json.load(f)["ours_30000"]
with open(os.path.join(args.ours_path, "per_view.json"), 'r') as f:
    ours_data = json.load(f)["ours_30000"]
N = len(base_data["PSNR"])
data = []
for i in range(N):
    base_psnr = base_data["PSNR"][f"{'%05d' % i}.png"]
    ours_psnr = ours_data["PSNR"][f"{'%05d' % i}.png"]
    base_ssim = base_data["SSIM"][f"{'%05d' % i}.png"]
    ours_ssim = ours_data["SSIM"][f"{'%05d' % i}.png"]
    base_lpips = base_data["LPIPS"][f"{'%05d' % i}.png"]
    ours_lpips = ours_data["LPIPS"][f"{'%05d' % i}.png"]
    if base_psnr < ours_psnr and (base_ssim - ours_ssim > args.threshold or ours_lpips - base_lpips > args.threshold):
        print(f"{'%05d' % i}.png")