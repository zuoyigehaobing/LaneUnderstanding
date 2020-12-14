"""
Model definitions in PyTorch

Logs:
    [12.09] Shan: initialized the skeleton and passed sanity check

To be discussed:
    - Freeze encoding layers?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchsummary


class SegNet(nn.Module):
    def __init__(self, in_channels=3, class_num=2, transfer_learning=True):
        super(SegNet, self).__init__()

        self.batch_norm_momentum = 0.1

        # Encoder: same as first 13 layers in VGG
        # Block1: Size = 1:1
        self.encoder_conv00 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv01 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Block2: Size = 2:1
        self.encoder_conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        # Block3: Size = 4:1
        self.encoder_conv20 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv21 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv22 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        # Block3: Size = 8:1
        self.encoder_conv30 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # Block4: Size = 16:1
        self.encoder_conv31 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv32 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # Block4: Size = 32:1
        self.encoder_conv40 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv41 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv42 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # Transfer learning from vgg16
        if transfer_learning:
            assert in_channels == 3
            self.vgg16 = models.vgg16(pretrained=True)
            self.encoder_conv00[0].weight = self.vgg16.features[0].weight
            self.encoder_conv01[0].weight = self.vgg16.features[2].weight
            self.encoder_conv10[0].weight = self.vgg16.features[5].weight
            self.encoder_conv11[0].weight = self.vgg16.features[7].weight
            self.encoder_conv20[0].weight = self.vgg16.features[10].weight
            self.encoder_conv21[0].weight = self.vgg16.features[12].weight
            self.encoder_conv22[0].weight = self.vgg16.features[14].weight
            self.encoder_conv30[0].weight = self.vgg16.features[17].weight
            self.encoder_conv31[0].weight = self.vgg16.features[19].weight
            self.encoder_conv32[0].weight = self.vgg16.features[21].weight
            self.encoder_conv40[0].weight = self.vgg16.features[24].weight
            self.encoder_conv41[0].weight = self.vgg16.features[26].weight
            self.encoder_conv42[0].weight = self.vgg16.features[28].weight

        # Decoder: I used conv2D here, since it's similar to conv2dtranspose
        self.decoder_conv42 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv41 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv40 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv32 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv31 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv30 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv22 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv21 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv20 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv10 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv01 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # no bn and relu before the final layer
        # Note: I didn't not apply softmax here or in the forward, since some
        #       loss functions such as crossentropy already included softmax
        #       inside
        self.decoder_conv00 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=class_num,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
                      )
        )

    def forward(self, x):

        # Encoder Stage1: remember the dimension and pooling indices
        x = self.encoder_conv00(x)
        x = self.encoder_conv01(x)
        dim_block1 = x.size()
        x, pool_idx1 = F.max_pool2d_with_indices(x,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 return_indices=True)

        # Encoder Stage2: remember the dimension and pooling indices
        x = self.encoder_conv10(x)
        x = self.encoder_conv11(x)
        dim_block2 = x.size()
        x, pool_idx2 = F.max_pool2d_with_indices(x,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 return_indices=True)

        # Encoder Stage3: remember the dimension and pooling indices
        x = self.encoder_conv20(x)
        x = self.encoder_conv21(x)
        x = self.encoder_conv22(x)
        dim_block3 = x.size()
        x, pool_idx3 = F.max_pool2d_with_indices(x,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 return_indices=True)

        # Encoder Stage4: remember the dimension and pooling indices
        x = self.encoder_conv30(x)
        x = self.encoder_conv31(x)
        x = self.encoder_conv32(x)
        dim_block4 = x.size()
        x, pool_idx4 = F.max_pool2d_with_indices(x,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 return_indices=True)

        # Encoder Stage5: remember the dimension and pooling indices
        x = self.encoder_conv40(x)
        x = self.encoder_conv41(x)
        x = self.encoder_conv42(x)
        dim_block5 = x.size()
        x, pool_idx5 = F.max_pool2d_with_indices(x,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 return_indices=True)

        # Decoder Stage 5: take the dimension from previous pooling layers
        x = F.max_unpool2d(x,
                           pool_idx5,
                           kernel_size=(2, 2),
                           stride=(2, 2),
                           output_size=dim_block5)
        x = self.decoder_conv42(x)
        x = self.decoder_conv41(x)
        x = self.decoder_conv40(x)

        # Decoder Stage 4: take the dimension from previous pooling layers
        x = F.max_unpool2d(x,
                           pool_idx4,
                           kernel_size=(2, 2),
                           stride=(2, 2),
                           output_size=dim_block4)
        x = self.decoder_conv32(x)
        x = self.decoder_conv31(x)
        x = self.decoder_conv30(x)

        # Decoder Stage 3: take the dimension from previous pooling layers
        x = F.max_unpool2d(x,
                           pool_idx3,
                           kernel_size=(2, 2),
                           stride=(2, 2),
                           output_size=dim_block3)
        x = self.decoder_conv22(x)
        x = self.decoder_conv21(x)
        x = self.decoder_conv20(x)

        # Decoder Stage 2: take the dimension from previous pooling layers
        x = F.max_unpool2d(x,
                           pool_idx2,
                           kernel_size=(2, 2),
                           stride=(2, 2),
                           output_size=dim_block2)
        x = self.decoder_conv11(x)
        x = self.decoder_conv10(x)

        # Decoder Stage 1: take the dimension from previous pooling layers
        x = F.max_unpool2d(x,
                           pool_idx1,
                           kernel_size=(2, 2),
                           stride=(2, 2),
                           output_size=dim_block1)
        x = self.decoder_conv01(x)
        x = self.decoder_conv00(x)

        # Note: again for convenience of the loss function, I didn't apply
        #       the soft-max function here

        return x


if __name__ == '__main__':

    # =============== sanity check ====================
    model = SegNet(3, 2)

    # randomize an array and do forward pass
    check_x = torch.zeros((20, 3, 224, 224))
    model.train()
    pred = model.forward(check_x)
    print(pred.shape)

    # report summary of the model
    print(torchsummary.summary(model, input_size=(3, 224, 224)))
