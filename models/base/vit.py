from models.base.modules import MLP
from models.funcational import *
from models.modules import _auto_pad, Ck1s1, ACT, Ck1s1NA

from utils.file import _pair


class FeedForwardResidual(MLP):
    def __init__(self, features, inner_features, dropout=0.0, act=ACT.GELU):
        super(FeedForwardResidual, self).__init__(
            in_features=features, out_features=features,
            inner_featuress=inner_features, dropout=dropout, act=act)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        out = self.norm(x)
        out = self.backbone(out)
        return out + x


class LocalAttentionMutiHead2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_head=8, qk_channels=64, dropout=0.0, kernel_size=7, dilation=1,
                 act=ACT.RELU):
        super().__init__()

        assert in_channels % num_head == 0
        assert out_channels % num_head == 0
        assert qk_channels % num_head == 0

        self.num_head = num_head
        self.qk_channels = qk_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.qkh_channels = qk_channels // num_head
        self.outh_channels = out_channels // num_head

        self.q = Ck1s1(in_channels=in_channels, out_channels=qk_channels, groups=num_head, bias=False)
        self.k = Ck1s1(in_channels=in_channels, out_channels=qk_channels, groups=num_head, bias=False)
        self.v = Ck1s1NA(in_channels=in_channels, out_channels=out_channels, groups=num_head, act=act)

        self.dropout = nn.Dropout(dropout)
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _auto_pad(self.kernel_size, self.dilation)
        self.num_token = np.prod(self.kernel_size)

    def forward(self, x):
        N, _, H, W = x.size()
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(N, self.num_head, self.qkh_channels, 1, H * W).permute(0, 1, 4, 3, 2)

        k_u = F.unfold(k, kernel_size=self.kernel_size, dilation=self.dilation, stride=1, padding=self.padding)
        v_u = F.unfold(v, kernel_size=self.kernel_size, dilation=self.dilation, stride=1, padding=self.padding)
        k_u = k_u.view(N, self.num_head, self.qkh_channels, self.num_token, H * W).permute(0, 1, 4, 2, 3)
        v_u = v_u.view(N, self.num_head, self.outh_channels, self.num_token, H * W).permute(0, 1, 4, 3, 2)

        dots = torch.matmul(q, k_u) * (self.qkh_channels ** -0.5)
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_u)
        out = out.permute(0, 1, 4, 2, 3).reshape(N, self.out_channels, H, W)
        return out


class LinearAttentionMutiHeadPtch(nn.Module):
    def __init__(self, in_features, out_features, qk_features, num_head=8, dropout=0.0, act=ACT.RELU):
        super().__init__()

        assert in_features % num_head == 0
        assert qk_features % num_head == 0
        assert out_features % num_head == 0

        self.num_head = num_head
        self.qk_features = qk_features
        self.in_features = in_features
        self.out_features = out_features

        self.qkh_features = qk_features // num_head
        self.outh_features = out_features // num_head

        self.q = nn.Linear(in_features, qk_features, bias=False)
        self.k = nn.Linear(in_features, qk_features, bias=False)
        self.v = nn.Linear(in_features, out_features, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = ACT.build(act)

    def forward(self, x):
        N, L, C = x.size()
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(N, L, self.num_head, self.qkh_features).permute(0, 2, 1, 3)
        k = k.view(N, L, self.num_head, self.qkh_features).permute(0, 2, 3, 1)
        v = v.view(N, L, self.num_head, self.outh_features).permute(0, 2, 1, 3)
        dots = torch.matmul(q, k) * (self.qkh_features ** -0.5)
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(N, L, self.out_features)
        return self.act(out)


class LinearAttentionResidual(LinearAttentionMutiHeadPtch):
    def __init__(self, features, num_head=8, qk_features=64, dropout=0.0, act=ACT.RELU):
        super().__init__(in_features=features, out_features=features, num_head=num_head, qk_features=qk_features,
                         dropout=dropout, act=act)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        out = self.norm(x)
        out = super(LinearAttentionResidual, self).forward(out)
        return out + x


class Transformer(nn.Module):
    def __init__(self, channels, mlp_channels=256, attn_channels=64, depth=6, num_head=8, dropout=0.0):
        super().__init__()
        backbone = []
        for _ in range(depth):
            backbone += [
                LinearAttentionResidual(features=channels, num_head=num_head, qk_features=attn_channels,
                                        dropout=dropout),
                FeedForwardResidual(features=channels, inner_features=mlp_channels, dropout=dropout),
            ]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ViT(nn.Module):
    def __init__(self, channels, img_size=(256, 256), patch_size=(32, 32), num_cls=20, depth=6, num_head=8,
                 mlp_channels=256, in_channels=3, attn_channels=64, dropout=0.0):
        super().__init__()
        W, H = img_size
        Wp, Hp = patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        assert W % Wp == 0 and H % Hp == 0, 'Image dimensions must be divisible by the patch size.'
        grid_size = (W // Wp, H // Hp)
        self.grid_size = grid_size

        num_patches = grid_size[0] * grid_size[1]
        patch_channels = in_channels * Hp * Wp
        self.projector = nn.Linear(patch_channels, channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(channels=channels, depth=depth, num_head=num_head,
                                       mlp_channels=mlp_channels, attn_channels=attn_channels, dropout=dropout)
        self.linear = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, num_cls)
        )

    def forward(self, img):
        Nb, C, H, W = img.size()
        Wp, Hp = self.patch_size
        Wg, Hg = self.grid_size
        img = img.view(Nb, C, Hg, Hp, Wg, Wp).permute(0, 2, 4, 1, 3, 5).reshape(Nb, Hg * Wg, C * Hp * Wp)
        x = self.projector(img)
        cls_tokens = self.cls_token.repeat(Nb, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(Hg * Wg + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

# if __name__ == "__main__":
#     v = ViT(
#         channels=1024, img_size=(256, 256), patch_size=(16, 16), num_cls=20, depth=2, num_head=8,
#         mlp_channels=256, in_channels=3, attn_channels=64, dropout=0.1,
#     )
#
#     img = torch.randn(2, 3, 256, 256)
#     preds = v(img)  # (1, 1000)
