import numpy as np
import torch
import librosa
import wav2clip
import clip

from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPLoss1D(nn.Module):
    def __init__(self, opts):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, image, audio):
        wav2clip_model = wav2clip.get_model()      
        image = self.avg_pool(self.upsample(image))

        text_features = torch.from_numpy(wav2clip.embed_audio(audio, wav2clip_model)).to(device).to(torch.float16)
        image_features = self.model.encode_image(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        similarity = 1 - logits_per_image / 100
        return similarity