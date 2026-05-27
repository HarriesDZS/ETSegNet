import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, meta_dim=None):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        # 仅在有 meta_dim 时注入 FiLM
        self.film = FiLM(out_ch, meta_dim) if meta_dim is not None else None

    def forward(self, x, meta=None):
        x = self.conv(x)
        if self.film is not None:
            x = self.film(x, meta)
        p = self.pool(x)
        return x, p

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class FiLM(nn.Module):
    """
    通过 meta 条件生成 scale 和 shift，调制卷积特征
    """
    def __init__(self, in_channels, meta_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(meta_dim, in_channels * 2),  # 输出 scale & shift
            nn.ReLU()
        )

    def forward(self, x, meta):
        """
        x: [B,C,H,W] feature map
        meta: [B, meta_dim] SUV 特征
        """
        gamma_beta = self.fc(meta)  # [B,2*C]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B,C] each
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        return x * gamma + beta


class PETBranch(nn.Module):

    def __init__(self, input_channel = 1, n_class = 2, base_ch=32, meta_dim=4):
        super().__init__()
        # Encoder: FiLM 注入
        self.down1 = Down(input_channel, base_ch, meta_dim=meta_dim)
        self.down2 = Down(base_ch, base_ch * 2, meta_dim=meta_dim)
        self.down3 = Down(base_ch * 2, base_ch * 4, meta_dim=meta_dim)
        # Bottleneck: 不注入 FiLM
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        # Decoder: 不注入 FiLM
        self.up3 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch)
        # Output
        self.out_conv = nn.Conv2d(base_ch, n_class-1, kernel_size=1)

    def forward(self, x, meta):
        # Encoder
        s1, p1 = self.down1(x, meta)
        s2, p2 = self.down2(p1, meta)
        s3, p3 = self.down3(p2, meta)
        # Bottleneck
        b = self.bottleneck(p3)
        # Decoder
        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        out = torch.sigmoid(self.out_conv(u1))
        return out


class CT_UNet_PET_Attention(nn.Module):
    def __init__(self, input_channel = 1, out_channle = 2, base_ch=32):
        super().__init__()
        # Encoder 使用复用的 Down
        self.down1 = Down(input_channel, base_ch)
        self.down2 = Down(base_ch, base_ch * 2)
        self.down3 = Down(base_ch * 2, base_ch * 4)
        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        # Decoder 使用复用的 Up
        self.up3 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch)
        # 输出
        self.out_conv = nn.Conv2d(base_ch, out_channle-1, 1)

    def forward(self, ct_img, pet_prob):
        """
        ct_img: [B,1,H,W] CT 图像
        pet_prob: [B,1,H,W] PET 分割预测概率图
        """
        # Encoder + PET 注意力
        s1, p1 = self.down1(ct_img)
        s1 = s1 * pet_prob

        s2, p2 = self.down2(p1)
        s2 = s2 * F.interpolate(pet_prob, size=s2.shape[2:], mode='bilinear', align_corners=False)

        s3, p3 = self.down3(p2)
        s3 = s3 * F.interpolate(pet_prob, size=s3.shape[2:], mode='bilinear', align_corners=False)

        # Bottleneck
        b = self.bottleneck(p3)
        b = b * F.interpolate(pet_prob, size=b.shape[2:], mode='bilinear', align_corners=False)

        # Decoder
        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)

        out = torch.sigmoid(self.out_conv(u1))
        return out


def normalize_tensor(x, eps=1e-8):
    """Per-sample min-max normalize to [0,1] along spatial dims (preserve batch & channel)."""
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    mn = x_flat.min(dim=2, keepdim=True)[0]
    mx = x_flat.max(dim=2, keepdim=True)[0]
    x_norm = (x_flat - mn) / (mx - mn + eps)
    return x_norm.view(B, C, H, W)

class LearnedEdgeMap(nn.Module):
    """
    Edge map extractor:
      - fixed Sobel kernels registered as buffers
      - a small learnable refinement CNN defined in __init__
    All learnable convs are created in __init__.
    """
    def __init__(self, refine_channels=8):
        super().__init__()
        # register fixed sobel kernels as buffers (shape [out_ch, in_ch, k, k])
        sobel_x = torch.tensor([[[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)  # [1,1,3,3]
        self.register_buffer("sobel_y", sobel_y)  # [1,1,3,3]

        # learnable refinement convs (defined in __init__)
        self.refine = nn.Sequential(
            nn.Conv2d(1, refine_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, p):
        """
        p: [B,1,H,W] probability map (expected in [0,1])
        returns:
            edge_out: [B,1,H,W] normalized edge map in [0,1]
        """
        # compute gradients via fixed Sobel (buffers)
        gx = F.conv2d(p, self.sobel_x, padding=1)
        gy = F.conv2d(p, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)  # [B,1,H,W]

        edge = normalize_tensor(grad_mag)                 # normalize per-sample
        edge_refined = self.refine(edge)                 # learnable refinement
        edge_out = normalize_tensor(edge_refined)        # final normalize
        return edge_out

class LearnedRoundnessMap(nn.Module):
    """
    Local roundness / compactness map:
      - fixed window kernel (ones) registered as buffer
      - fixed Sobel kernels registered as buffers
      - learnable refinement convs defined in __init__
    All learnable convs are created in __init__.
    """
    def __init__(self, kernel_size=15, refine_channels=8):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size

        # register ones window kernel as buffer for local sums
        ones = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        self.register_buffer("window_kernel", ones)  # [1,1,k,k]

        # register fixed sobel kernels as buffers
        sobel_x = torch.tensor([[[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # learnable refinement convs (defined in __init__)
        self.refine = nn.Sequential(
            nn.Conv2d(1, refine_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, p):
        """
        p: [B,1,H,W] probability map (expected in [0,1])
        returns:
            round_out: [B,1,H,W] normalized local roundness in [0,1]
        """
        B, C, H, W = p.shape
        assert C == 1, "LearnedRoundnessMap expects single-channel probability map"

        # soft mask is p itself
        soft = p

        # compute local area via convolution with ones kernel (buffer)
        padding = self.kernel_size // 2
        local_area = F.conv2d(soft, self.window_kernel, padding=padding)  # [B,1,H,W]

        # compute local gradient magnitude (per-pixel)
        gx = F.conv2d(soft, self.sobel_x, padding=1)
        gy = F.conv2d(soft, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-8)  # [B,1,H,W]

        # compute local perimeter as local sum of gradient magnitude
        local_perimeter = F.conv2d(grad, self.window_kernel, padding=padding)  # [B,1,H,W]

        # compute local roundness: 4*pi*A / (P^2)
        local_round = 4.0 * math.pi * local_area / (local_perimeter * local_perimeter + 1e-8)
        local_round = torch.relu(local_round)  # clamp negative

        # normalize and refine (learnable)
        round_norm = normalize_tensor(local_round)
        round_refined = self.refine(round_norm)
        round_out = normalize_tensor(round_refined)
        return round_out

def compute_edge_clarity(ct_prob, eps=1e-8):
    """
    基于 CT 分割预测概率图生成边缘清晰度特征图
    可微，可梯度回传
    假设每张图只有一个肿瘤区域，肿瘤内部值相同，背景为0
    输入:
        ct_prob: [B,1,H,W] CT 分割预测概率图
    输出:
        edge_clarity_map: [B,1,H,W] 边缘清晰度概率图
    """
    # --------------------------
    # 1. 计算每个像素熵
    # H(p) = -p*log(p) - (1-p)*log(1-p)
    # --------------------------
    entropy_map = - ct_prob * torch.log(ct_prob + eps) - (1 - ct_prob) * torch.log(1 - ct_prob + eps)

    # --------------------------
    # 2. 计算可微边缘权重 (Sobel)
    # --------------------------
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0).to(ct_prob.device)
    sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0).to(ct_prob.device)

    grad_x = F.conv2d(ct_prob, sobel_x, padding=1)
    grad_y = F.conv2d(ct_prob, sobel_y, padding=1)
    edge_prob = torch.sqrt(grad_x**2 + grad_y**2 + eps)

    # --------------------------
    # 3. 生成 soft 肿瘤 mask 保持可微
    # --------------------------
    # sigmoid 近似硬阈值
    soft_mask = torch.sigmoid((ct_prob - 0.5) * 20)  # 斜率可调，越大越接近硬阈值

    # --------------------------
    # 4. 计算肿瘤边缘平均熵
    # --------------------------
    numerator = (entropy_map * edge_prob * soft_mask).sum(dim=[2,3], keepdim=True)
    denominator = (edge_prob * soft_mask).sum(dim=[2,3], keepdim=True) + eps
    edge_clarity = numerator / denominator  # [B,1,1,1]

    # --------------------------
    # 5. 广播到肿瘤区域
    # --------------------------
    edge_clarity_map = edge_clarity * soft_mask  # 背景趋近0，肿瘤区域值相同

    return edge_clarity_map


def compute_roundness(ct_prob, eps=1e-8):
    """
    基于 CT 分割预测概率图计算肿瘤形状规整度（Roundness）
    可微，可梯度回传
    假设图像中只有一个肿瘤区域，肿瘤区域值相同，背景为0
    输入:
        ct_prob: [B,1,H,W] CT 分割预测概率图
    输出:
        roundness_map: [B,1,H,W] 形状规整度概率图，肿瘤区域值相同，背景为0
    """
    # --------------------------
    # 1. 生成可微软肿瘤 mask
    # --------------------------
    soft_mask = torch.sigmoid((ct_prob - 0.5) * 20)  # soft mask 保持可微

    # --------------------------
    # 2. 计算面积
    # --------------------------
    area = soft_mask.sum(dim=[2,3], keepdim=True)  # [B,1,1,1]

    # --------------------------
    # 3. 计算边界长度（周长）通过 Sobel 梯度
    # --------------------------
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0).to(ct_prob.device)
    sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0).to(ct_prob.device)

    grad_x = F.conv2d(soft_mask, sobel_x, padding=1)
    grad_y = F.conv2d(soft_mask, sobel_y, padding=1)
    perimeter = torch.sqrt(grad_x**2 + grad_y**2 + eps).sum(dim=[2,3], keepdim=True)  # [B,1,1,1]

    # --------------------------
    # 4. 计算 Roundness
    # --------------------------
    roundness = 4 * math.pi * area / (perimeter**2 + eps)  # [B,1,1,1]

    # --------------------------
    # 5. 广播到肿瘤区域
    # --------------------------
    roundness_map = roundness * soft_mask  # 背景趋近0，肿瘤区域值相同

    return roundness_map

class PET_Fusion_Pixelwise_Attention(nn.Module):
    """
    将 PET 分割概率图、边缘清晰度图、形状规整度图像素级加权融合
    每个像素点都有自己的注意力权重
    输入:
        pet_prob: [B,1,H,W]
        edge_map: [B,1,H,W]
        round_map: [B,1,H,W]
    输出:
        fused_prob: [B,1,H,W] 最终融合分割概率
        attn_map: [B,3,H,W] 每个像素点对应三个概率图的注意力权重
    """

    def __init__(self, in_channels=3, hidden_ch=16):
        super().__init__()
        # 简单 1x1 conv 网络生成像素级注意力
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, kernel_size=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.refine =  nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, pet_prob, edge_map, round_map):
        # 1. 拼接三个特征图
        x = torch.cat([pet_prob, edge_map, round_map], dim=1)  # [B,3,H,W]

        # 2. 生成像素级注意力 logits
        attn_logits = self.attn_conv(x)  # [B,3,H,W]
        # 3. softmax 在通道维度归一化，每个像素点的三个权重和为1
        #attn_map = F.softmax(attn_logits, dim=1)  # [B,3,H,W]
        attn_map = attn_logits

        #print(x.shape, attn_map.shape)


        fused_prob = x * attn_map
        fused_prob = self.refine(fused_prob)

        return fused_prob, attn_map

class PET_Fusion_Pixelwise_AttentionV2(nn.Module):
    """
    将 PET 分割概率图、边缘清晰度图、形状规整度图像素级加权融合
    每个像素点都有自己的注意力权重
    输入:
        pet_prob: [B,1,H,W]
        edge_map: [B,1,H,W]
        round_map: [B,1,H,W]
    输出:
        fused_prob: [B,1,H,W] 最终融合分割概率
        attn_map: [B,3,H,W] 每个像素点对应三个概率图的注意力权重
    """

    def __init__(self, in_channels=3, hidden_ch=16):
        super().__init__()
        # 简单 1x1 conv 网络生成像素级注意力
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, pet_prob, edge_map, round_map):
        # 1. 拼接三个特征图
        x = torch.cat([pet_prob, edge_map, round_map], dim=1)  # [B,3,H,W]

        # 2. 生成像素级注意力 logits
        attn_logits = self.attn_conv(x)  # [B,3,H,W]


        fused_prob = torch.sigmoid(attn_logits)

        return fused_prob, attn_logits


class PET_Fusion_Transformer_Attention(nn.Module):
    """
    Transformer 风格像素级注意力融合
    输入:
        pet_prob: [B,1,H,W]
        edge_map: [B,1,H,W]
        round_map: [B,1,H,W]
    输出:
        fused_prob: [B,1,H,W]
        attn_weights: [B,3,H,W]
    """

    def __init__(self, in_channels=3, embed_dim=16, num_heads=1):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv2d(embed_dim, 1, kernel_size=1)  # 融合成最终概率

    def forward(self, pet_prob, edge_map, round_map):
        B, _, H, W = pet_prob.shape
        # 1. 拼接三个特征图
        x = torch.cat([pet_prob, edge_map, round_map], dim=1)  # [B,3,H,W]

        # 2. 通道 embedding
        x_emb = self.embed(x)  # [B, embed_dim, H, W]

        # 3. reshape 为序列形式 (每个像素为一个 token)
        x_seq = x_emb.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # [B, H*W, embed_dim]

        # 4. Transformer self-attention
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # [B, H*W, embed_dim]

        # 5. reshape 回特征图
        attn_feat = attn_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, embed_dim, H, W]

        # 6. 投影到最终分割概率
        fused_prob = torch.sigmoid(self.proj(attn_feat))  # [B,1,H,W]

        # 7. 如果需要像素级 attention 权重，可用 softmax 映射到三个输入通道
        #    这里可以在 embedding 前对三个输入做一个小 MLP + softmax
        attn_logits = nn.Conv2d(3, 3, kernel_size=1).to(pet_prob.device)(x)
        attn_weights = F.softmax(attn_logits, dim=1)  # [B,3,H,W]

        return fused_prob, attn_weights

class ExplainMode(nn.Module):
    def __init__(self, input_channel = 1, out_class = 2):
        super().__init__()
        self.pet_branch = PETBranch(input_channel=input_channel, n_class=out_class)
        self.ct_branch = CT_UNet_PET_Attention(input_channel=input_channel, out_channle=out_class)
        self.fusion = PET_Fusion_Pixelwise_Attention()
        #self.edge = LearnedEdgeMap()
        #self.roudn = LearnedRoundnessMap()
        '''
        self.edge = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 1, kernel_size=3, padding=1)
        )
        self.roudn = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 1, kernel_size=3, padding=1)
        )
        '''

    def forward(self, ct, pet, meta):
        pet_output = self.pet_branch(pet, meta)
        ct_output = self.ct_branch(ct, pet_output)
        #edge_map = self.edge(ct_output)
        #round_map = self.roudn(ct_output)
        edge_map = compute_edge_clarity(ct_output)
        round_map = compute_roundness(ct_output)
        fusion_seg,atten_map = self.fusion(pet_output, edge_map, round_map)
        return pet_output,ct_output, fusion_seg, atten_map


class ExplainModeForShow(nn.Module):
    def __init__(self, input_channel = 1, out_class = 2):
        super().__init__()
        self.pet_branch = PETBranch(input_channel=input_channel, n_class=out_class)
        self.ct_branch = CT_UNet_PET_Attention(input_channel=input_channel, out_channle=out_class)
        self.fusion = PET_Fusion_Pixelwise_Attention()
        #self.edge = LearnedEdgeMap()
        #self.roudn = LearnedRoundnessMap()
        '''
        self.edge = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 1, kernel_size=3, padding=1)
        )
        self.roudn = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 1, kernel_size=3, padding=1)
        )
        '''

    def forward(self, ct, pet, meta):
        pet_output = self.pet_branch(pet, meta)
        ct_output = self.ct_branch(ct, pet_output)
        #edge_map = self.edge(ct_output)
        #round_map = self.roudn(ct_output)
        edge_map = compute_edge_clarity(ct_output)
        round_map = compute_roundness(ct_output)
        fusion_seg,atten_map = self.fusion(pet_output, edge_map, round_map)
        return edge_map,round_map,ct_output, pet_output, fusion_seg, atten_map


if __name__ == '__main__':
    model = ExplainMode()
    pet = torch.randn((1, 1, 512, 512))
    ct = torch.randn((1, 1, 512, 512))
    meta = torch.randn((1, 4))
    pet_output, ct_output, fusion_seg, atten_map = model(ct, pet, meta)
    print(pet_output.shape, ct_output.shape, fusion_seg.shape, atten_map.shape)