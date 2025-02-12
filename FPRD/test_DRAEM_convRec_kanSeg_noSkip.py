import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_convRec_kan_Seg_noSkip import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import cv2


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):

    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'
        
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()
        

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)


        mask_cnt = 0

        anomaly_score_gt = []

        
        pic_i=0
        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()


            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            image_np = gray_rec.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            # print("image_np",image_np)
            cv2.imwrite(f"/home/lh/deeplearning/kan_beifen/rec/{pic_i}.png",image_np*255)
            
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            
            image_np = out_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            gray_image = image_np[:, :, 0]


            gray_image = np.expand_dims(gray_image, axis=2)
            ###################
            gray_image[gray_image>0]=1
            ###################
            cv2.imwrite(f"/home/lh/deeplearning/kan_beifen/seg/{pic_i}.png",gray_image*255)
            gray_batch_np = gray_batch.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(f"/home/lh/deeplearning/kan_beifen/ori/{pic_i}.png",gray_batch_np*255)
            pic_i +=1
            mask_cnt += 1


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    obj_list = ['capsule',
                 'bottle',
                 'carpet',
                 'leather',
                 'pill',
                 'transistor',
                 'tile',
                 'cable',
                 'zipper',
                 'toothbrush',
                 'metal_nut',
                 'hazelnut',
                 'screw',
                 'grid',
                 'wood'
                 ]

    obj_list = [

                  "wood"

                 ]
    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name)
