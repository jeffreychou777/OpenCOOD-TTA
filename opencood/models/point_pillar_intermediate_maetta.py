# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import copy

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.a41_blocks import spatial_mask
from opencood.tools import train_utils
from opencood.models.sub_modules.a41_blocks import RecDecoder
from opencood.models.sub_modules.a41_blocks import AutoEncoder


class PointPillarIntermediateMaetta(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateMaetta, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)
        self.backbone = train_utils.load_pretrained_model(
            args["saved_pth"], self.backbone)
        self.Autoencoder = AutoEncoder()

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 3, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}
            


        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict_masked = copy.deepcopy(batch_dict)
        
        # shape of spatial_features:[11,64,200,704]
        batch_dict_masked['spatial_features'] = spatial_mask(batch_dict_masked['spatial_features'], 0.5, 'random') 
        
        
        batch_dict = self.backbone(batch_dict)
        batch_dict_masked = self.backbone(batch_dict_masked)
        spatial_features_2d = batch_dict['spatial_features_2d']
        spatial_features_2d_masked = batch_dict_masked['spatial_features_2d']
        spatial_features_2d_rec = self.Autoencoder(spatial_features_2d_masked)


        # psm = self.cls_head(spatial_features_2d)
        # rm = self.reg_head(spatial_features_2d)

        output_dict = {'fused_feature_org': spatial_features_2d,
                       'fused_feature_rec': spatial_features_2d_rec,
                       }
        
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})
            
        return output_dict