from models.rcnn.maskrcnn import *


class PANet(MaskRCNN):
    def __init__(self, backbone, classifier, masker, device=None, pack=None):
        super(PANet, self).__init__(backbone=backbone, classifier=classifier, masker=masker,
                                    device=device, pack=pack)

    @staticmethod
    def merge_pool(featmaps, xyxys, ids_b, img_size, feat_size=(7, 7)):
        ids_xyxys = torch.cat([ids_b[:, None], xyxys], dim=1)
        areas_feat = torch.Tensor([featmap.size(2) * featmap.size(3) for featmap in featmaps])
        area_img = img_size[0] * img_size[1]
        spatial_scales = list(torch.sqrt(areas_feat / area_img))
        featmaps_roi = []
        for i, (featmap, spatial_scale) in enumerate(zip(featmaps, spatial_scales)):
            featmap_roi = torchvision.ops.roi_align(
                input=featmap, boxes=ids_xyxys, output_size=feat_size, spatial_scale=spatial_scale)
            featmaps_roi.append(featmap_roi)
        featmaps_roi = torch.cat(featmaps_roi, dim=1)
        return featmaps_roi

    def apply_classifier(self, featmaps, xyxys, ids_b):
        featmaps_roi = PANet.merge_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                        feat_size=self.feat_size_clfr)
        rois_regd = self.pkd_modules['classifier'](featmaps_roi, xyxysT2xywhsT(xyxys))
        return rois_regd

    def apply_masker(self, featmaps, xyxys, ids_b):
        featmaps_roi = PANet.merge_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                        feat_size=self.feat_size_mskr)
        maskss = self.pkd_modules['masker'](featmaps_roi)
        return maskss

    @staticmethod
    def ResNetR18(num_cls=20, img_size=(512, 512), device=None, pack=None, anchor_ratio=8):
        backbone = PANetMutiScaleResNetBkbn.R18(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R18['channels']
        num_layer = len(backbone.layers)
        classifier = PANetMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, num_layer=num_layer)
        masker = PANetRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH,
                                       num_layer=num_layer)
        return PANet(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def ResNetR50(num_cls=20, img_size=(512, 512), device=None, pack=None, anchor_ratio=8):
        backbone = PANetMutiScaleResNetBkbn.R50(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R50['channels']
        num_layer = len(backbone.layers)
        classifier = PANetMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, num_layer=num_layer)
        masker = PANetRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH,
                                       num_layer=num_layer)
        return PANet(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def ResNetR101(num_cls=20, img_size=(512, 512), device=None, pack=None, anchor_ratio=8):
        backbone = PANetMutiScaleResNetBkbn.R101(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R101['channels']
        num_layer = len(backbone.layers)
        classifier = PANetMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, num_layer=num_layer)
        masker = PANetRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH,
                                       num_layer=num_layer)
        return PANet(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def Const(num_cls=20, img_size=(512, 512), device=None, batch_size=1, anchor_ratio=8):
        channels = (num_cls + 1) * 5 + num_cls
        backbone = MaskRCNNConstBkbn(img_size=img_size, batch_size=batch_size, out_channels=channels,
                                     anchor_ratio=anchor_ratio)
        num_layer = len(backbone.layers)
        classifier = PANetConstClassifier(num_cls=num_cls, in_channels=channels, num_layer=num_layer)
        masker = PANetConstMasker(num_cls=num_cls, in_channels=channels, num_layer=num_layer)
        return PANet(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = PANet.ResNetR18(img_size=(800, 800), num_cls=20, device='cpu')
    # model.img_size = (256, 256)
    # model.export_onnx(onnx_dir='/home/user/JD/Public/export', prifix='mskr101', batch_size_bkbn=1,
    #                   batch_size_inner=1)
