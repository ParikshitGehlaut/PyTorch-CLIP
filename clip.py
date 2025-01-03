import torch 
from torch import nn
from torch.nn import functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        pos_embedding = self.pos_embed.repeat(B, 1, 1)
        return pos_embedding
    
    
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.classifier(x[:, 0])
    

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio,
            n_classes,
            dropout,
            device
            ):
        
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device))
        self.pos_embed = PositionalEmbedding(self.patch_embed.n_patches, embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.classification_head = ClassificationHead(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed(x)
        x = self.encoder(x)
        x = self.classification_head(x)
        return x


class TextTransformer(nn.Module):
    def __init__(
            self,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio, 
            vocab_size,
            dropout,
            device
            ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + 512, embed_dim, device=device))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def forward(self, input_ids):
        # input_ids : (B, L)
        _, L = input_ids.shape

        tok_embedding = self.token_embedding(input_ids)
        pos_embedding = self.positional_embedding[:, :L, :]
        x = tok_embedding + pos_embedding

        x = self.encoder(x)

        return x[:, 0]


class CLIP(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size, 
            in_channels,
            embed_dim,
            n_heads,
            n_layers, 
            mlp_ratio,
            vocab_size,
            device,
            dropout=0.2
    ):
        super().__init__()

        self.vision_transformer = VisionTransformer(
                img_size,
                patch_size,
                in_channels,
                embed_dim,
                n_heads,
                n_layers,
                mlp_ratio,
                n_classes=embed_dim,
                dropout=dropout,
            )
        
        self.text_transformer = TextTransformer(
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio, 
            vocab_size,
            dropout,
            device
        )

    def forward(self, image, input_ids):
        image_features = self.vision_transformer(image)
        text_features = self.text_transformer(input_ids)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return image_features, text_features

