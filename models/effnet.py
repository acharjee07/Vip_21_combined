import os
import torch
import torch.nn as nn
import timm


class Modelx(nn.Module):
    
    def __init__(self, model_name, pretrained=True):
        
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.bn_out = self.model.bn2._parameters['weight'].shape[0]
        self.conv_xtra = nn.Conv2d(self.bn_out, 2048, 1)         


    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.blocks[0](x)

        x1 = self.model.blocks[1](x)

        x2 = self.model.blocks[2](x1)

        x3 = self.model.blocks[3](x2)

        x4 = self.model.blocks[4](x3)

        x5 = self.model.blocks[5](x4)

        x6 = self.model.blocks[6](x5)

        x = self.model.conv_head(x6)
        x = self.model.bn2(x)
        x = self.conv_xtra(x)
        return x, (x2, x3, x4, x5, x6)


mx = Modelx('efficientnet_b3')
mx(torch.zeros(1, 3, 256, 256))[0].shape




class Modely(nn.Module):
    def __init__(self):
        super(Modely, self).__init__()
        self.inplanes = 2048
        self.res_mode = False
        self.BN_MOMENTUM = 0.1
        self.deconv_with_bias = True
        self.dconv_dict = self.get_dconv_config()
        
        self.dconv_layer_1 = self._make_deconv_layer(0)
        self.dconv_layer_2 = self._make_deconv_layer(1)        
        self.dconv_layer_3 = self._make_deconv_layer(2)
        self.dconv_layer_4 = self._make_deconv_layer(3)      
        self.dconv_layer_5 = self._make_deconv_layer(4)
        self.dconv_layer_6 = self._make_deconv_layer(5)

#         self.res_converter_1 = nn.Conv2d(728, 256, 2, padding = 1)
#         self.res_converter_2 = nn.Conv2d(256, 256, 2, padding = 2)
        
        
        self.final_layer = nn.Sequential(
                                nn.Conv2d(
                                in_channels = self.dconv_dict['out_channel'][-1],
                                out_channels = 14,
                                kernel_size= 1,
                                stride= 1,
                                padding= 0
                                ),
                                nn.Sigmoid(),
                            )

    def get_dconv_config(self):
        return {
            'n_dconv' : 6,
            'kernels' : [1, 1, 2, 1, 2, 2],
            'strides' : [1, 1, 2, 1, 2, 2],
            'padding' : [0, 0, 0, 0, 0, 0],
            'out_padding' : [0, 0, 0, 0, 0, 0],
            'in_channel' : [self.inplanes, 384*2, 232*2, 136*2, 96*2, 48*2] if self.res_mode else [self.inplanes, 384, 232, 136, 96, 48],
            'out_channel' : [384, 232, 136, 96, 48, 32]
        }


    def _make_deconv_layer(self, i):
        
        layers = []
        kernel = self.dconv_dict['kernels'][i]
        padding = self.dconv_dict['padding'][i]
        output_padding = self.dconv_dict['out_padding'][i]
        stride = self.dconv_dict['strides'][i]


        in_plane = self.dconv_dict['in_channel'][i]
        out_plane = self.dconv_dict['out_channel'][i]

        layers.append(
            nn.ConvTranspose2d(
                in_channels = in_plane,
                out_channels = out_plane,
                kernel_size = kernel,
                stride = stride,
                padding = padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
        layers.append(nn.BatchNorm2d(out_plane, momentum=self.BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)

    def forward(self, x, xs):
        x1, x2, x3, x4, x5 = xs

        x = self.dconv_layer_1(x) 

        if self.res_mode:
            x = torch.cat((x, x5), dim = 1)
        x = self.dconv_layer_2(x)

        if self.res_mode:
            x = torch.cat((x, x4), dim = 1)
        x = self.dconv_layer_3(x)

        if self.res_mode:
            x = torch.cat((x, x3), dim = 1)
        x = self.dconv_layer_4(x)

        if self.res_mode:
            x = torch.cat((x, x2), dim = 1)
        x = self.dconv_layer_5(x)

        if self.res_mode:
            x = torch.cat((x, x1), dim = 1)
        x = self.dconv_layer_6(x)
#         print(x.shape)

        x = self.final_layer(x)

        return x

my = Modely()




class Modelxy(nn.Module):
    def __init__(self,model_name='resnet101'):
        super(Modelxy, self).__init__()
        self.model_name=model_name
        self.modelx=Modelx(model_name=self.model_name)
        self.modely=Modely()

    def forward(self,x):
        x, xs =self.modelx(x)
        x=self.modely(x, xs)

        return x
    
    def save(self,optim,epoch):
        self.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': 0,
            }, 'modelxy{}.pth'.format(epoch))
    def load(self,optim,path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        

model_name='efficientnet_b0'
mxy = Modelxy(model_name)
print(mxy(torch.zeros(1, 3, 288, 288)).shape)