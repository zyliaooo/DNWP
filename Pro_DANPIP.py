"""
Demo for DANPIP
"""

import torch
from Pro_clc import *
from Pro_masknet import *
from Pro_R import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class DANPIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.clc = torch.load('./XX/XX')
        self.masknet = torch.load('./XX/XX').module
        self.probenet = torch.load('./XX/XX').module
        self.bfanet = torch.load('./XX/XX').module
    def forward(self, x, mag):
        pred_np = x.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
        plt.imshow(pred_np[:, :, 30], cmap='gray')
        plt.colorbar()
        plt.show()

        # print(x.shape)
        _, _, h, w, d = x.shape
        clc_input = x[:, :, 16:-16, 16:-16, 30]
        clc_input = torch.where(clc_input > clc_input.min(), clc_input, torch.zeros_like(clc_input))  # Correction background
        score = torch.sigmoid(self.clc(clc_input))
        print(f'the Tag prob of the phase need to be masked is {score.item():.2f}')
        if score > 0.5:
            mask = torch.sigmoid(self.masknet(mag))
            mask = torch.where(mask > 0.5, 1.0, 0.0)    # 0.3 0.5 0.7
        x = torch.mul(x, mask)

        pred_np = x.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
        plt.imshow(pred_np[:, :, 30], cmap='gray')
        plt.colorbar()
        plt.show()

        out = self.probenet(x)

        pred_np = out.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
        plt.imshow(pred_np[:, :, 30], cmap='gray')
        plt.colorbar()
        plt.show()

        complexity = (out - x) / (2 * math.pi)
        complexity = complexity.max() - complexity.min() + 1
        if complexity < 4:
            return out
        print('Activate BFANet')
        out = self.bfanet(x)

        pred_np = out.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
        plt.imshow(pred_np[:, :, 30], cmap='gray')
        plt.colorbar()
        plt.show()

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data_path = r"./XX/XX"
pre_image = scio.loadmat(test_data_path)
phase = pre_image['data'][:,:,:96]
mag = pre_image['mag'][:,:,:96]

phase = torch.from_numpy(phase).unsqueeze(0)
phase = phase.type(torch.FloatTensor).unsqueeze(1).cuda()

mag = torch.from_numpy(mag).unsqueeze(0)
mag = mag.type(torch.FloatTensor).unsqueeze(1).cuda()
model = DANPIP().cuda()
with torch.no_grad():
    model.eval()
    out = model(phase, mag)
