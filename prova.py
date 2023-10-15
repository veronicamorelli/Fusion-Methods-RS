# Pansharpening example

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# im 40

def import_toarray(path, im):
    path = path + im
    im_raster = rasterio.open(path)
    return np.moveaxis(im_raster.read(), 0, -1) 

path_rgb = "/home/veronica/Scrivania/RSIm/Dataset/Train/RGB/"
path_dem = "/home/veronica/Scrivania/RSIm/Dataset/Train/DEM/"
path_pan = "/home/veronica/Scrivania/RSIm/Dataset/Train/PAN/PAN/"
path_vnir = "/home/veronica/Scrivania/RSIm/Dataset/Train/VNIR/"
path_swir = "/home/veronica/Scrivania/RSIm/Dataset/Train/SWIR/SWIR/"
path_hs_pan = "/home/veronica/Scrivania/RSIm/Dataset/Train/HS/"

im = "40.tif"
im_np = "40.npy"

rgb = import_toarray(path_rgb, im)
dem = import_toarray(path_dem, im)
pan = import_toarray(path_pan, im)
vnir = import_toarray(path_vnir, im)
swir = import_toarray(path_swir, im)
hs_pan = np.load(path_hs_pan+im_np)

print(rgb.shape, dem.shape, pan.shape, vnir.shape, swir.shape, hs_pan.shape)

vnir = np.expand_dims(vnir[:, :, vnir.shape[2]//2], 2)
swir = np.expand_dims(swir[:, :, swir.shape[2]//2], 2)

# plt.figure(1)
plt.imshow(rgb)
plt.show()


# segm_crit : loss
# calcolo una loss per ogni output (2 output in outputs)
# loss = segm_crit(soft_output, target) --> task-specific loss for the m-th modality
for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print('train input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample['mask'].cuda().long()
        # Compute outputs
        outputs, masks = segmenter(inputs)
        loss = 0
        for output in outputs:
            output = nn.functional.interpolate(output, size=target.size()[1:],
                                               mode='bilinear', align_corners=False)
            soft_output = nn.LogSoftmax()(output)
            # Compute loss and backpropagate
            loss += segm_crit(soft_output, target)

        if lamda > 0:
            L1_loss = 0
            for mask in masks:
                L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
            loss += lamda * L1_loss
        
        optimizer.zero_grad()
        loss.backward()
        if print_loss:
            print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)