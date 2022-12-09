import torch
from torch import nn  #nn means neural network
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from torchsummary import summary
import PIL
import torchvision.transforms as T


def pair(t):
    return t if isinstance(t,tuple) else (t,t)   # isinstance() returns True if the specified object is of the specified type


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)
 
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=100, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
            ) if project_out else nn.Identity()
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        #get qkv tuple:([batch, No_patch, No_heads * head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)  #torch.chunk(inputs, chunks, dim=) split a tensor into the specified number of chunks
        
        #split q into [batch, No_heads, No.patch, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  #shape of q/k/v [batch, heads, sequence_len, dim]
        
        
        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  #k.transpose(-1,-2)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        
        #out = torch.matmul(attn, v)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        #call all output -> [batch, No_patch, NO_head*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout=0., emb_dropout=0. ):
        super().__init__()
        image_height, image_width = pair(image_size)  #input image size 50*50
        patch_height, patch_width = pair(patch_size)  #patch size: the No. pixels per patch
        
        # assert statement checks if a condition is True, program keep running if True
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image dimensions must be divisible by the patch size'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls','mean'}, 'pool type is either cls token or mean pooling'
        
        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim), # project patch dimension to Transformer Encoder required dimension
            )
        
        #positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
            )
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding    #[:, :(n+1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:,0]
        
        #x = self.to_latent(x)
        return self.mlp_head(x)
    

