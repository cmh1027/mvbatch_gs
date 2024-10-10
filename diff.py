import numpy as np
from PIL import Image
x = np.array(Image.open("offset0.png")).astype(np.float32)
y = np.array(Image.open("offset100.png")).astype(np.float32)
diff50 = x[:, :-50] - y[:, 50:]
Image.fromarray(np.abs(diff50.mean(axis=-1)).astype(np.uint8)).save("diff50.png")
diff100 = x[:, :-100] - y[:, 100:]
Image.fromarray(np.abs(diff100.mean(axis=-1)).astype(np.uint8)).save("diff100.png")