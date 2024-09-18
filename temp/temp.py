from PIL import Image
import numpy as np
a = np.array(Image.open("00028_part.png"))
Image.fromarray(a[:, :3114//2]).save("00028_part_.png")
print('ehllo')