from models.base.vgg import VGGMain, RevVGGMain, VGGBkbn, DSAMP_TYPE, RevVGGBkbn
from models.modules import *
from models.template import GAN, GANVAE, WGAN, VAE, AE
from utils import *


class Discriminator(VGGBkbn):
    def __init__(self, repeat_nums, channelss, strides, act=ACT.RELU, norm=NORM.BATCH, out_channels=1,
                 in_channels=3, dsamp_type=DSAMP_TYPE.MAX):
        super(Discriminator, self).__init__(repeat_nums=repeat_nums, channelss=channelss, strides=strides,
                                            act=act, norm=norm, in_channels=in_channels, dsmp_type=dsamp_type)
        self.cvtor = Ck1s1(in_channels=channelss[-1], out_channels=out_channels)

    def forward(self, imgs):
        feats = super(Discriminator, self).forward(imgs)
        return self.cvtor(feats)

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, in_channels=3, out_channels=1, ):
        return Discriminator(**VGGBkbn.PARA_NANO, act=act, in_channels=in_channels,
                             out_channels=out_channels, norm=norm)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, in_channels=3, out_channels=1, ):
        return Discriminator(**VGGBkbn.PARA_SMALL, act=act, in_channels=in_channels,
                             out_channels=out_channels, norm=norm)


class SimpleGAN(GAN):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
              generate_repeat=3):
        generator = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                     out_channels=in_channels, pool_size=pool_size)
        discriminator = Discriminator.Nano(in_channels=in_channels, act=ACT.LK,
                                           norm=NORM.BATCH, out_channels=1)
        return SimpleGAN(generator=generator, discriminator=discriminator, device=device, pack=pack,
                         features=features, img_size=img_size, generate_repeat=generate_repeat)


class SimpleWGAN(WGAN):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
              generate_repeat=3):
        generator = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                     out_channels=in_channels, pool_size=pool_size)
        discriminator = VGGMain.Nano(img_size=img_size, in_channels=in_channels, pool_size=pool_size, act=ACT.LK,
                                     norm=NORM.BATCH, num_cls=1)
        return SimpleWGAN(generator=generator, discriminator=discriminator, device=device, pack=pack,
                          features=features, img_size=img_size, generate_repeat=generate_repeat)

    @staticmethod
    def Medium(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
               generate_repeat=3):
        generator = RevVGGMain.Medium(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                      out_channels=in_channels, pool_size=pool_size)
        discriminator = VGGMain.Small(img_size=img_size, in_channels=in_channels, pool_size=pool_size, act=ACT.LK,
                                      norm=NORM.BATCH, num_cls=1)
        return SimpleWGAN(generator=generator, discriminator=discriminator, device=device, pack=pack,
                          features=features, img_size=img_size, generate_repeat=generate_repeat)


class SimpleSNGAN(WGAN):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
              generate_repeat=3):
        generator = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                     out_channels=in_channels, pool_size=pool_size)
        discriminator = VGGMain.Nano(img_size=img_size, in_channels=in_channels, pool_size=pool_size, act=ACT.LK,
                                     norm=NORM.BATCH, num_cls=1)
        return SimpleSNGAN(generator=generator, discriminator=discriminator, device=device, pack=pack,
                           features=features, img_size=img_size, generate_repeat=generate_repeat)

    @staticmethod
    def Medium(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
               generate_repeat=3):
        generator = RevVGGMain.Medium(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                      out_channels=in_channels, pool_size=pool_size)
        discriminator = VGGMain.Nano(img_size=img_size, in_channels=in_channels, pool_size=pool_size, act=ACT.LK,
                                     norm=NORM.BATCH, num_cls=1)
        return SimpleSNGAN(generator=generator, discriminator=discriminator, device=device, pack=pack,
                           features=features, img_size=img_size, generate_repeat=generate_repeat)


class SimpleGANVAE(GANVAE):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None,
              generate_repeat=1):
        encoder = VGGMain.Small(in_channels=in_channels, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                num_cls=features * 2, pool_size=pool_size)
        decoder = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                   out_channels=in_channels, pool_size=pool_size)
        discriminator = VGGMain.Nano(img_size=img_size, in_channels=in_channels, pool_size=pool_size, act=ACT.LK,
                                     norm=NORM.BATCH, num_cls=1)
        return SimpleGANVAE(encoder=encoder, decoder=decoder, discriminator=discriminator, device=device, pack=pack,
                            features=features, img_size=img_size, generate_repeat=generate_repeat)


class SimpleVAE(VAE):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None, ):
        encoder = VGGMain.Small(in_channels=in_channels, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                num_cls=features * 2, pool_size=pool_size)
        decoder = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                   out_channels=in_channels, pool_size=pool_size)
        return SimpleVAE(encoder=encoder, decoder=decoder, device=device, pack=pack,
                         features=features, img_size=img_size)


class SimpleAE(AE):

    @staticmethod
    def Small(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None, ):
        encoder = VGGMain.Small(in_channels=in_channels, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                num_cls=features, pool_size=pool_size)
        decoder = RevVGGMain.Small(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                   out_channels=in_channels, pool_size=pool_size)
        return SimpleAE(encoder=encoder, decoder=decoder, device=device, pack=pack,
                        img_size=img_size)

    @staticmethod
    def Medium(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None, ):
        encoder = VGGMain.Medium(in_channels=in_channels, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                 num_cls=features, pool_size=pool_size)
        decoder = RevVGGMain.Medium(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                    out_channels=in_channels, pool_size=pool_size)
        return SimpleAE(encoder=encoder, decoder=decoder, device=device, pack=pack,
                        img_size=img_size)

    @staticmethod
    def MediumChan(device=None, pack=None, img_size=(224, 224), in_channels=3,):
        encoder = VGGBkbn.Medium(in_channels=in_channels, act=ACT.LK, norm=NORM.BATCH)
        decoder = RevVGGBkbn.Medium(in_channels=256, act=ACT.LK, norm=NORM.BATCH, out_channels=in_channels)
        return SimpleAE(encoder=encoder, decoder=decoder, device=device, pack=pack,
                        img_size=img_size)

    @staticmethod
    def Large(device=None, pack=None, features=50, img_size=(224, 224), in_channels=3, pool_size=None, ):
        encoder = VGGMain.Large(in_channels=in_channels, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                num_cls=features, pool_size=pool_size)
        decoder = RevVGGMain.Large(in_features=features, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                   out_channels=in_channels, pool_size=pool_size)
        return SimpleAE(encoder=encoder, decoder=decoder, device=device, pack=pack,
                        img_size=img_size)


if __name__ == '__main__':
    model = SimpleWGAN.Medium(device=0, pack=PACK.AUTO, features=256, generate_repeat=1, pool_size=(4, 4))
    model.generator.export_onnx('../../gen')
    model.discriminator.export_onnx('../../dis')

# if __name__ == '__main__':
#
#     model = SimpleAE.Medium(img_size=(64, 64), pool_size=(4, 4))
#     model.decoder.export_onnx('../../dec')
#     model.encoder.export_onnx('../../enc')

# if __name__ == '__main__':
#     model = SimpleGAN.Example()
#     noise = torch.zeros(size=(1, 50, 5, 5)).to(model.device)
#     fimgs = model.generator(noise)
#
#     y = model.discriminator(fimgs)
#     model.save('./buff')
#     print(y.size())
