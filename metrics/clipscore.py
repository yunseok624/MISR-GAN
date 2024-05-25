import clip
import torch
import open_clip
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY

# def three_grayscale(img):
#     img_list = [img.clone() for _ in range(3)]
#     three = torch.cat(img_list, dim=0)
#     return three

# @METRIC_REGISTRY.register()
# def calculate_clipscore(img, img2, clip_model, **kwargs):
#     device = torch.device('cuda')

#     if clip_model == 'clip-ViT-B/16':
#         model, _ = clip.load("ViT-B/16", device=device)
#         img_size = (224,224)
#     elif clip_model == 'clipa-ViT-bigG-14':
#         model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
#         model = model.to(device)
#         img_size = (336,336)
#     elif clip_model == 'siglip-ViT-SO400M-14':
#         model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
#         model = model.to(device)
#         if len(img.shape) == 3:
#             img_size = (384,384)
#         else:
#             img_size = (192, 192)
#     else:
#         print(clip_model, " is not supported for CLIPScore.")

#     if len(img.shape) == 3:
#         tensor1 = torch.as_tensor(img).unsqueeze(2).permute(2, 0, 1)
#         tensor1 = tensor1.unsqueeze(0).to(device).float()/255
#         tensor2 = torch.as_tensor(img2).unsqueeze(2).permute(2, 0, 1)
#         tensor2 = tensor2.unsqueeze(0).to(device).float()/255
        
#         tensor1 = F.interpolate(tensor1, img_size)
#         tensor2 = F.interpolate(tensor2, img_size)
#     else:
#         tensor1 = three_grayscale(torch.as_tensor(img).unsqueeze(0))
#         tensor1 = tensor1.to(device).float()/255
#         tensor2 = three_grayscale(torch.as_tensor(img2).unsqueeze(0))
#         tensor2 = tensor2.to(device).float()/255
        
#         tensor1 = F.interpolate(tensor1, img_size)
#         tensor2 = F.interpolate(tensor2, img_size)

#     feats1 = model.encode_image(tensor1)
#     feats2 = model.encode_image(tensor2)

#     clip_score = F.cosine_similarity(feats1, feats2).detach().item()
#     return clip_score

@METRIC_REGISTRY.register()
def calculate_clipscore(img, img2, clip_model, **kwargs):
    device = torch.device('cuda')

    if clip_model == 'clip-ViT-B/16':
        model, _ = clip.load("ViT-B/16", device=device)
        img_size = (224,224)
    elif clip_model == 'clipa-ViT-bigG-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
        model = model.to(device)
        img_size = (336,336)
    elif clip_model == 'siglip-ViT-SO400M-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
        model = model.to(device)
        img_size = (384,384)
    else:
        print(clip_model, " is not supported for CLIPScore.")

    tensor1 = torch.as_tensor(img).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    tensor1 = F.interpolate(tensor1, img_size)
    tensor2 = F.interpolate(tensor2, img_size)

    feats1 = model.encode_image(tensor1)
    feats2 = model.encode_image(tensor2)

    clip_score = F.cosine_similarity(feats1, feats2).detach().item()
    return clip_score