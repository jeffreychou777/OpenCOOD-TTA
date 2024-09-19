import torch
import torch.nn as nn

class RecDecoder(nn.Module):
    def __init__(self, feature_num, layer_num):
        super().__init__()
        self.feature_num = feature_num
        self.feature_stride = 2

        self.decoder_rec = nn.ModuleList()

        # reconstruction decoder
        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [nn.Sequential(
                nn.ConvTranspose2d(
                    feature_num, feature_num,
                    kernel_size=2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(feature_num,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]

            cur_layers.extend([nn.Sequential(
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=2, bias=False, padding=1
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3,
                               momentum=0.01),
                nn.ReLU()
            )])
            self.decoder_rec.append(nn.Sequential(*cur_layers)).to('cuda')
            feature_num //= 2

    def forward(self, x):
        # reconstruction
        for i in range(len(self.decoder_rec)-1, -1, -1):
            x_rec = self.decoder_rec[i](x)

        return x_rec

class AutoEncoder(nn.Module):
    def __init__(self, feature_num, layer_num):
        super().__init__()
        self.feature_num = feature_num
        self.feature_stride = 2

        self.encoder = nn.ModuleList()

        self.decoder_tsk = nn.ModuleList()
        self.decoder_rec = nn.ModuleList()

        for i in range(layer_num):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=2, padding=0, bias=False
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()]

            cur_layers.extend([
                nn.Conv2d(feature_num, feature_num // self.feature_stride,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_num // self.feature_stride,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])

            self.encoder.append(nn.Sequential(*cur_layers)).to('cuda')
            feature_num = feature_num // self.feature_stride

        # task decoder
        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [nn.Sequential(
                nn.ConvTranspose2d(
                    feature_num // 2, feature_num,
                    kernel_size=2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(feature_num,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]

            cur_layers.extend([nn.Sequential(
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=1, bias=False, padding=1
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3,
                               momentum=0.01),
                nn.ReLU()
            )])
            self.decoder_tsk.append(nn.Sequential(*cur_layers)).to('cuda')
            feature_num //= 2

        # reconstruction decoder
        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [nn.Sequential(
                nn.ConvTranspose2d(
                    feature_num // 2, feature_num,
                    kernel_size=2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(feature_num,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]

            cur_layers.extend([nn.Sequential(
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=1, bias=False, padding=1
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3,
                               momentum=0.01),
                nn.ReLU()
            )])
            self.decoder_rec.append(nn.Sequential(*cur_layers)).to('cuda')
            feature_num //= 2

    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        # reconstruction
        x_rec = x
        for i in range(len(self.decoder_rec)-1, -1, -1):
            x_rec = self.decoder_rec[i](x_rec)

        return x_rec

def spatial_mask(feature, mask_ratio, mask_type='random'):
    """
    mask some points in the feature map.
    :param feature: [B, C, H, W]
    :return: feature_masked: [B, C, H, W]
    """
    mask_ratio = mask_ratio

    # mask = activate_mask_generator(sampled_feat[i], sample_pixels)
    if mask_type == 'random':
        mask = random_mask_generator(feature=feature, mask_ratio=mask_ratio)
        feature_masked = feature * mask
    if mask_type == 'activate':
        feature_masked = activate_feature_generator(feature=feature, mask_ratio=mask_ratio)
    
    return feature_masked

def random_mask_generator(feature, mask_ratio):
    mask_ratio = mask_ratio
    N, C, H, W = feature.shape
    random_mask = torch.rand((N, C, H, W), device=feature.device) > mask_ratio
    
    return random_mask

def activate_feature_generator(feature, mask_ratio):  # Need to be modified
    '''
    input: x_updated: [N, C, H, W]
    '''
    N, C, H, W = feature.shape

    activate_masks = torch.zeros((N, H, W), device=feature.device)
    topk = int(H * W * (1- mask_ratio))
    feature_masked = feature.clone()
    for i, f in enumerate(feature):
        # add all channels
        f_add = torch.sum(f, dim=0) 
        # reshape
        f_temp = f_add.reshape(-1) 
        # initialize mask
        mask = torch.zeros_like(f_temp)  
        # sort by descend
        x_sort = torch.sort(f_temp, descending=True)

        mask[x_sort.indices[:topk]] = 1

        feature_masked[i] = f * mask.reshape(1, H, W)
    
    return feature_masked
