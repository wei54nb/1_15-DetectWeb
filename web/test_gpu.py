from torchvision.ops import nms
import torch

boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
indices = nms(boxes, scores, 0.5)
print(indices)
