"""
Functional testing
"""
import skimage
import math
import torch
from Pro_masknet import *     # Pro_clc   Pro_masknet    Pro_R
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_data_path = r"./XX/XX"
model_name = 'XX'
pred_tag = 'XX'  # phase or mask or classify
save_tag = False
model_path = './models/' + model_name


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path).cuda()
    print(model)

    with torch.no_grad():
        model.eval()
        if pred_tag == 'mask':
            pre_image = scio.loadmat(test_data_path)
            mag = pre_image['mag']
            # mag = pre_image['mag'][..., n:n+96]          # for clip
            # mag = np.pad(mag, ((0, 0), (0, 0), (0, 2)), 'constant')   # for pad
            # mag = mag / np.max(mag)      # if need
            print(mag.shape)

            mag = torch.from_numpy(mag).unsqueeze(0)
            mag = mag.type(torch.FloatTensor).unsqueeze(1).cuda()
            pred = model(mag)

            # pred = pred[..., :-2]     # for pad
            pred = torch.sigmoid(pred)
            pred = torch.where(pred > 0.5, 1.0, 0.0)    # Suggest adjusted in [0.3, 0.7]
            # ->cpu->numpy
            pred_np = pred.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
            plt.imshow(pred_np[:, :, 30], cmap='gray')
            plt.colorbar()
            plt.show()

        elif pred_tag == 'phase':
            pre_image = scio.loadmat(test_data_path)
            data = pre_image['data']
            data = np.pad(data, ((0, 0), (0, 0), (0, 2)), 'constant')   #  for pad
            # data = pre_image['data'][..., n:n+96]      # for clip
            print(data.shape)

            # to tensor
            data = torch.from_numpy(data).unsqueeze(0)
            data = data.type(torch.FloatTensor).unsqueeze(1).cuda()
            pred = model(data)
            pred = pred[..., :-2]  # for pad

            # ->cpu->numpy
            pred_np = pred.squeeze().type(torch.FloatTensor).cpu().detach().numpy()
            plt.imshow(pred_np[:, :, 30], cmap='gray')
            plt.colorbar()
            plt.show()

        elif pred_tag == 'classify':
            pre_image = scio.loadmat(test_data_path)
            phase = pre_image['phase']
            phase_min = phase.min()
            phase = np.where(phase > phase_min, phase, 0)   # Correction background
            plt.figure(figsize=(8, 8), dpi=600)
            plt.imshow(phase, cmap='gray')
            plt.axis('off')
            plt.show()

            phase = torch.from_numpy(phase).unsqueeze(0)
            phase = phase.type(torch.FloatTensor).unsqueeze(1).cuda()
            pred = model(phase)
            pred = torch.sigmoid(pred)
            print(f'the Tag score of the phase should be masked is {pred.item():.2f}')

        # print
        print("------------------")
        print("Done!")
        print("------------------")

        # save results
        if save_tag:
            if pred_tag == 'mask':
                scio.savemat(test_data_path.replace(".mat", f"_result({model_name}).mat"),
                             {
                                 'pred_mask': pred_np
                             })
            elif pred_tag == 'phase':
                scio.savemat(test_data_path.replace(".mat", f"_result({model_name}).mat"),
                             {
                                 'pred2': pred_np
                             })
