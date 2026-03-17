import cv2
import numpy as np

OUTPUT_FILE = "output.mp4"

# load the image and convert it to grayscale
image = cv2.imread("images/me.jpg", cv2.IMREAD_GRAYSCALE)

# compute the compact SVD
U, Sigma, Vh = np.linalg.svd(image)
print(U.shape)
print(Sigma.shape)
print(Vh.shape)

print("Compiling Video")

# setup the video writer
height, width = image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 24, (width, height), isColor=False)

for n in range(1, 150):
    # compute the rank-n approximation
    frame = np.clip(U[:, :n] @ np.diag(Sigma[:n]) @ Vh[:n, :], 0, 255).astype(np.uint8)

    # append the frame
    out.write(frame)

    print(f"Completed frame {n}")

out.release()
print(f"Video saved to {OUTPUT_FILE}")


