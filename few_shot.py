import torch
from data.AnomalyShapeNet import Dataset3dad_ShapeNet_train_memory
from utils import utils
import numpy as np
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np


obj_list = [
            'ashtray0','bag0',
            'bottle0','bottle1','bottle3','bowl0','bowl1','bowl2','bowl3','bowl4','bowl5','bucket0','bucket1','cap0','cap3','cap4','cap5',
            'cup0','cup1','eraser0','headset0','headset1','helmet0','helmet1','helmet2','helmet3','jar0','microphone0','shelf0',
            'tap0','tap1','vase0','vase1','vase2','vase3','vase4','vase5','vase7','vase8','vase9',
            ]

def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, dataset_name, device):
    mem_features = {}
    features_list = [3,7,11]
    for obj in obj_list:
        data = Dataset3dad_ShapeNet_train_memory(dataset_dir, obj, 1024, True)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

        for data, mask, label, path in dataloader:
            basic_template = data.squeeze(0).cpu().numpy()
            break


        features = []
        for pc, mask, label, sample_path in dataloader:
            pc = pc.cuda().to(torch.float32)

            with torch.no_grad():

                reg_data = get_registration_np(pc.squeeze(0).cpu().numpy().astype(np.float64),basic_template)
                reg_data = torch.tensor(np.expand_dims(reg_data, axis=0)).to(torch.float32).to(device)

                pc_features_m, patch_tokens_m, center_idx_m = utils.get_model(model).encode_pc(reg_data)
                patch_tokens_m = [patch_tokens_m[j] for j in features_list]

                patch_tokens_m = [p[0, 1:, :] for p in patch_tokens_m]
                features.append(patch_tokens_m)
            

        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features
