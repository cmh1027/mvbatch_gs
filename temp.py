import time
import torch
s = time.time()
for _ in range(30000):
    x = torch.ones(200, device=torch.device('cuda'))
    torch.multinomial(x, 4, replacement=True)
print(time.time() - s)