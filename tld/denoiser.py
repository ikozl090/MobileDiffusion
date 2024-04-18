"""transformer based denoiser"""

import torch
from einops.layers.torch import Rearrange
from torch import nn
from diffusers import UNet2DConditionModel 
from tld.configs import DataDownloadConfig, DataConfig, ModelConfig, TrainConfig, DenoiserConfig

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size / self.patch_size) * (self.img_size / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                self.n_channels,
                patch_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=int(self.img_size / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    mlp_class=MLPSepConv,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)


# class Denoiser(nn.Module):
#     def __init__(
#         self,
#         image_size: int,
#         noise_embed_dims: int,
#         patch_size: int,
#         embed_dim: int,
#         dropout: float,
#         n_layers: int,
#         text_emb_size: int = 768,
#         mlp_multiplier: int = 4,
#         n_channels: int = 4
#     ):
#         super().__init__()

#         self.image_size = image_size
#         self.noise_embed_dims = noise_embed_dims
#         self.embed_dim = embed_dim
#         self.n_channels = n_channels

#         self.fourier_feats = nn.Sequential(
#             SinusoidalEmbedding(embedding_dims=noise_embed_dims),
#             nn.Linear(noise_embed_dims, self.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )

#         self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
#         self.norm = nn.LayerNorm(self.embed_dim)
#         self.label_proj = nn.Linear(text_emb_size, self.embed_dim)

#     def forward(self, x, noise_level, label):
#         noise_level = self.fourier_feats(noise_level).unsqueeze(1)

#         label = self.label_proj(label).unsqueeze(1)
        
#         print(noise_level.shape)
#         print(label.shape)

#         noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
#         noise_label_emb = self.norm(noise_label_emb)

#         x = self.denoiser_trans_block(x, noise_label_emb)

#         return x

class Denoiser(UNet2DConditionModel): 
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        text_emb_size: int = 768,
        mlp_multiplier: int = 4,
        n_channels: int = 4
    ): 
        # super(Denoiser,self).__init__(
        #     sample_size=image_size,  # the target image resolution
        #     time_embedding_dim = DenoiserConfig.text_emb_size,
        #     cross_attention_dim = DenoiserConfig.text_emb_size,
        #     in_channels=4,  # the number of input channels, 3 for RGB images
        #     out_channels=4,  # the number of output channels
        #     layers_per_block=2,  # how many ResNet layers to use per UNet block
        #     block_out_channels=(40, 80, 160, 320),  # the number of output channels for each UNet block
        #     only_cross_attention=True, 
        #     mid_block_only_cross_attention=True,
        #     norm_num_groups=10,
        #     down_block_types=(
        #         "DownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
        #         "AttnDownBlock2D",
        #     ),
        #     up_block_types=(
        #         "AttnUpBlock2D",  # a regular ResNet upsampling block
        #         "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
        #         "CrossAttnUpBlock2D",
        #         "UpBlock2D",
        #     ),
        #     mid_block_type="UNetMidBlock2DCrossAttn",
        # )
        
        # super(Denoiser,self).__init__(
        #     sample_size=image_size,  # the target image resolution
        #     time_embedding_dim = DenoiserConfig.text_emb_size,
        #     cross_attention_dim = DenoiserConfig.text_emb_size,
        #     in_channels=4,  # the number of input channels, 3 for RGB images
        #     out_channels=4,  # the number of output channels
        #     layers_per_block=2,  # how many ResNet layers to use per UNet block
        #     block_out_channels=(80, 320, 640, 1280),  # the number of output channels for each UNet block
        #     only_cross_attention=True, 
        #     mid_block_only_cross_attention=True,
        #     down_block_types=(
        #         "DownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
        #         "AttnDownBlock2D",
        #     ),
        #     up_block_types=(
        #         "AttnUpBlock2D",  # a regular ResNet upsampling block
        #         "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
        #         "CrossAttnUpBlock2D",
        #         "UpBlock2D",
        #     ),
        #     mid_block_type="UNetMidBlock2DCrossAttn",
        # )
        
        super(Denoiser,self).__init__(
            sample_size=image_size,  # the target image resolution
            time_embedding_dim = DenoiserConfig.text_emb_size,
            cross_attention_dim = DenoiserConfig.text_emb_size,
            in_channels=4,  # the number of input channels, 3 for RGB images
            out_channels=4,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(160, 320, 640, 1080),  # the number of output channels for each UNet block
            only_cross_attention=True, 
            mid_block_only_cross_attention=True,
            norm_num_groups=40,
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
        )