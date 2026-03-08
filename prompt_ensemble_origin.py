import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from utils import utils


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    def norm(embedding):
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.mean(dim=0)
        embedding /= embedding.norm()
        return embedding
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['bulge {}', 'sink {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a point cloud of a {}.', 'a point cloud of the {}.', 'point cloud of a  {}.', 'point cloud of a big {}.', 'point cloud depth map of a {}.', 'a point cloud of the {}.', 'a point cloud of a {}.','there is a {} point cloud in the scene.']

    text_prompts = {
    }
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = utils.get_model(model).encode_text(prompted_sentence)
            text_features.append(norm(class_embeddings))

        text_features = torch.stack(text_features, dim=1).to(device)
        
        text_prompts[obj] = text_features
    return text_prompts
