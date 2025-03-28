import torch
import torch.nn as nn
import numpy as np
from functools import partial
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper
from ldm.util import default
import clip


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class FaceClipEncoder(AbstractEncoder):
    def __init__(self, augment=True, retreival_key=None):
        super().__init__()
        self.encoder = FrozenCLIPImageEmbedder()
        self.augment = augment
        self.retreival_key = retreival_key

    def forward(self, img):
        encodings = []
        with torch.no_grad():
            x_offset = 125
            if self.retreival_key:
                face = img[:,3:,190:440,x_offset:(512-x_offset)]
                other = img[:,:3,...].clone()
            else:
                face = img[:,:,190:440,x_offset:(512-x_offset)]
                other = img.clone()

            if self.augment:
                face = K.RandomHorizontalFlip()(face)

            other[:,:,190:440,x_offset:(512-x_offset)] *= 0
            encodings = [
                self.encoder.encode(face),
                self.encoder.encode(other),
            ]

        return torch.cat(encodings, dim=1)

    def encode(self, img):
        if isinstance(img, list):
            return torch.zeros((1, 2, 768), device=self.encoder.model.visual.conv1.weight.device)

        return self(img)


class FaceIdClipEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenCLIPImageEmbedder()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.id = FrozenFaceEncoder(".../stable-diffusion/model_ir_se50.pth", augment=True)

    def forward(self, img):
        encodings = []
        with torch.no_grad():
            face = kornia.geometry.resize(img, (256, 256), interpolation='bilinear', align_corners=True)

            other = img.clone()
            other[:,:,184:452,122:396] *= 0
            encodings = [
                self.id.encode(face),
                self.encoder.encode(other),
            ]

        return torch.cat(encodings, dim=1)

    def encode(self, img):
        if isinstance(img, list):
            return torch.zeros((1, 2, 768), device=self.encoder.model.visual.conv1.weight.device)

        return self(img)


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        return self(text)


from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel


def disabled_train(self, mode=True):
    return self


class FrozenT5Embedder(AbstractEncoder):
    def __init__(self, version="t5-v1_1-large", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version, cache_dir='')
        self.transformer = T5EncoderModel.from_pretrained(version, cache_dir='')
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


from ldm.thirdp.psp.id_loss import IDFeatures
import kornia.augmentation as K


class FrozenFaceEncoder(AbstractEncoder):
    def __init__(self, model_path, augment=False):
        super().__init__()
        self.loss_fn = IDFeatures(model_path)
        for p in self.loss_fn.parameters():
            p.requires_grad = False
        self.mapper = torch.nn.Linear(512, 768)
        p = 0.25
        if augment:
            self.augment = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomEqualize(p=p))
        else:
            self.augment = False

    def forward(self, img):
        if isinstance(img, list):
            return torch.zeros((1, 1, 768), device=self.mapper.weight.device)

        if self.augment is not None:
            img = self.augment((img + 1)/2)
            img = 2*img - 1

        feat = self.loss_fn(img, crop=True)
        feat = self.mapper(feat.unsqueeze(1))
        return feat

    def encode(self, img):
        return self(img)


class FrozenCLIPEmbedder(AbstractEncoder):
    def __init__(self, version="clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version, cache_dir='')
        self.transformer = CLIPTextModel.from_pretrained(version, cache_dir='')
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


import torch.nn.functional as F
from transformers import CLIPVisionModel
class ClipImageProjector(AbstractEncoder):


    def __init__(self, version="clip-vit-large-patch14", max_length=77):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(version)
        self.model.train()
        self.max_length = max_length
        self.antialias = True
        self.mapper = torch.nn.Linear(1024, 768)
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        null_cond = self.get_null_cond(version, max_length)
        self.register_buffer('null_cond', null_cond)

    @torch.no_grad()
    def get_null_cond(self, version, max_length):
        device = self.mean.device
        embedder = FrozenCLIPEmbedder(version=version, device=device, max_length=max_length)
        null_cond = embedder([""])
        return null_cond

    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        if isinstance(x, list):
            return self.null_cond
        x = self.preprocess(x)
        outputs = self.model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.mapper(last_hidden_state)
        return F.pad(last_hidden_state, [0,0, 0,self.max_length-last_hidden_state.shape[1], 0,0])

    def encode(self, im):
        return self(im)


class ProjectedFrozenCLIPEmbedder(AbstractEncoder):
    def __init__(self, version="clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.embedder = FrozenCLIPEmbedder(version=version, device=device, max_length=max_length)
        self.projection = torch.nn.Linear(768, 768)

    def forward(self, text):
        z = self.embedder(text)
        return self.projection(z)

    def encode(self, text):
        return self(text)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)
        del self.model.transformer
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        if isinstance(x, list):
            device = self.model.visual.conv1.weight.device
            return torch.zeros(1, 768, device=device)
        return self.model.encode_image(self.preprocess(x)).float()

    def encode(self, im):
        return self(im).unsqueeze(1)


from torchvision import transforms
import random


class FrozenCLIPImageMutliEmbedder(AbstractEncoder):
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cpu',
            antialias=True,
            max_crops=5,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)
        del self.model.transformer
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.max_crops = max_crops

    def preprocess(self, x):
        randcrop = transforms.RandomResizedCrop(224, scale=(0.085, 1.0), ratio=(1,1))
        max_crops = self.max_crops
        patches = []
        crops = [randcrop(x) for _ in range(max_crops)]
        patches.extend(crops)
        x = torch.cat(patches, dim=0)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        if isinstance(x, list):
            device = self.model.visual.conv1.weight.device
            return torch.zeros(1, self.max_crops, 768, device=device)
        batch_tokens = []
        for im in x:
            patches = self.preprocess(im.unsqueeze(0))
            tokens = self.model.encode_image(patches).float()
            for t in tokens:
                if random.random() < 0.1:
                    t *= 0
            batch_tokens.append(tokens.unsqueeze(0))

        return torch.cat(batch_tokens, dim=0)

    def encode(self, im):
        return self(im)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like


class LowScaleEncoder(nn.Module):
    def __init__(self, model_config, linear_start, linear_end, timesteps=1000, max_noise_level=250, output_size=64,
                 scale_factor=1.0):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x):
        z = self.model.encode(x).sample()
        z = z * self.scale_factor
        noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


if __name__ == "__main__":
    from ldm.util import count_params
    sentences = [""]
    model = FrozenT5Embedder(version="").cuda()
    count_params(model, True)
    z = model(sentences)
    print(z.shape)

    model = FrozenCLIPEmbedder().cuda()
    count_params(model, True)
    z = model(sentences)
    print(z.shape)

    print("done.")
