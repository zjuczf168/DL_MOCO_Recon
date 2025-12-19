"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from fastmri.models import VarNet, VarNetMidc, MocoSimLayer, UndersampleLayer

from .mri_module import MriModule

class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        maxTrans: int = 10, 
        maxRot: float = 10.0, 
        N_tp_Max: float = 16.0, 
        half_width_no_motion: int = 12, 
        noMotionFrac: float = 0.5,
        center_fraction: float = 0.08, 
        acceleration: float = 4.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.maxTrans = maxTrans 
        self.maxRot = maxRot
        self.N_tp_Max = N_tp_Max 
        self.half_width_no_motion = half_width_no_motion
        self.noMotionFrac = noMotionFrac
        self.center_fraction = center_fraction
        self.acceleration = acceleration

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )
        
        self.mocoSimUnd = MocoSimLayer(maxTrans=self.maxTrans, maxRot=self.maxRot, N_tp_Max=self.N_tp_Max, half_width_no_motion=self.half_width_no_motion, noMotionFrac=self.noMotionFrac)
        
        self.und = UndersampleLayer(center_fraction=self.center_fraction, acceleration=self.acceleration)

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.varnet(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        kspace, _, target, _, _, max_value, _ = batch
                      
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        
        # --- varnet recon --- #    
        outksimgList = self(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
        
        # loss_acc = []
        # for output in outksimgList:
            # target, output = transforms.center_crop_to_smallest(target, output)
            # loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
            # loss_acc.append(loss)
        # loss = sum(loss_acc)/len(outksimgList)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        kspace, _, target, fname, slice_num, max_value, _ = batch
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd.forward(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und.forward(kspace_sim)
        # --- varnet recon --- #   
        outksimgList = self.forward(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        kspace, _, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        # --- varnet recon --- #   
        outksimgList = self(masked_kspace, mask)
        output = outksimgList[-1]

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        
        parser.add_argument(
            "--maxTrans",
            default=10.0,
            type=float,
            help="max translation in pixel for motion simulation",
        )
        
        parser.add_argument(
            "--maxRot",
            default=10.0,
            type=float,
            help="max rotation in degrees for motion simulation",
        )
        
        parser.add_argument(
            "--N_tp_Max",
            default=16,
            type=int,
            help="max number of motion events",
        )
        
        parser.add_argument(
            "--half_width_no_motion",
            default=12,
            type=int,
            help="half width around the center of k-space with no motion event",
        )
        
        parser.add_argument(
            "--noMotionFrac",
            default=0.25,
            type=float,
            help="Fraction of samples without any motion simulation",
        )
        
        parser.add_argument(
            "--center_fraction",
            default=0.08,
            type=float,
            help="Fully sampled Center fraction for undersampling",
        )
        
        parser.add_argument(
            "--acceleration",
            default=4.0,
            type=float,
            help="Undersampling factor",
        )

        return parser


class VarNetModuleMi(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        maxTrans: int = 10, 
        maxRot: float = 10.0, 
        N_tp_Max: float = 16.0, 
        half_width_no_motion: int = 12, 
        noMotionFrac: float = 0.5,
        center_fraction: float = 0.08, 
        acceleration: float = 4.0,
        variable_center: float = 0.2,
        center_lines: int = 0, 
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.maxTrans = maxTrans 
        self.maxRot = maxRot
        self.N_tp_Max = N_tp_Max 
        self.half_width_no_motion = half_width_no_motion
        self.noMotionFrac = noMotionFrac
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.center_lines = center_lines
        self.variable_center = variable_center

        self.varnetmi = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )
        
        self.mocoSimUnd = MocoSimLayer(maxTrans=self.maxTrans, maxRot=self.maxRot, N_tp_Max=self.N_tp_Max, half_width_no_motion=self.half_width_no_motion, noMotionFrac=self.noMotionFrac)
        
        self.und = UndersampleLayer(center_fraction=self.center_fraction, acceleration=self.acceleration, center_lines=self.center_lines, variable_center=self.variable_center)

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.varnetmi(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        kspace, _, target, _, _, max_value, _ = batch
                      
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        
        # --- varnet recon --- #    
        outksimgList = self(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
        
        # loss_acc = []
        # for output in outksimgList:
            # target, output = transforms.center_crop_to_smallest(target, output)
            # loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
            # loss_acc.append(loss)
        # loss = sum(loss_acc)/len(outksimgList)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        kspace, _, target, fname, slice_num, max_value, _ = batch
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd.forward(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und.forward(kspace_sim)
        # --- varnet recon --- #   
        outksimgList = self.forward(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        kspace, _, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        # --- varnet recon --- #   
        outksimgList = self(masked_kspace, mask)
        output = outksimgList[-1]

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        
        parser.add_argument(
            "--maxTrans",
            default=10.0,
            type=float,
            help="max translation in pixel for motion simulation",
        )
        
        parser.add_argument(
            "--maxRot",
            default=10.0,
            type=float,
            help="max rotation in degrees for motion simulation",
        )
        
        parser.add_argument(
            "--N_tp_Max",
            default=16,
            type=int,
            help="max number of motion events",
        )
        
        parser.add_argument(
            "--half_width_no_motion",
            default=12,
            type=int,
            help="half width around the center of k-space with no motion event",
        )
        
        parser.add_argument(
            "--noMotionFrac",
            default=0.25,
            type=float,
            help="Fraction of samples without any motion simulation",
        )
        
        parser.add_argument(
            "--center_fraction",
            default=0.08,
            type=float,
            help="Fully sampled Center fraction for undersampling",
        )
        
        parser.add_argument(
            "--variable_center",
            default=0.2,
            type=float,
            help="percentage of fully sampled center to vary",
        )
        
        parser.add_argument(
            "--center_lines",
            default=0,
            type=int,
            help="number of fixed center lines",
        )
        
        parser.add_argument(
            "--acceleration",
            default=4.0,
            type=float,
            help="Undersampling factor",
        )

        return parser


class VarNetModuleMidc(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        maxTrans: int = 10, 
        maxRot: float = 10.0, 
        N_tp_Max: float = 16.0, 
        half_width_no_motion: int = 12, 
        noMotionFrac: float = 0.5,
        center_fraction: float = 0.08, 
        acceleration: float = 4.0,
        variable_center: float = 0.2,
        center_lines: int = 0, 
        dc_chans: int = 8,
        dc_pools: int = 4,
        midc_w: int = 320, 
        midc_h: int = 320,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.maxTrans = maxTrans 
        self.maxRot = maxRot
        self.N_tp_Max = N_tp_Max 
        self.half_width_no_motion = half_width_no_motion
        self.noMotionFrac = noMotionFrac
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.center_lines = center_lines
        self.variable_center = variable_center
        self.dc_chans = dc_chans
        self.dc_pools = dc_pools
        self.midc_w = midc_w 
        self.midc_h = midc_h

        self.varnetmidc = VarNetMidc(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
            dc_chans=self.dc_chans,
            dc_pools=self.dc_pools,
            midc_w=self.midc_w,
            midc_h=self.midc_h,
        )
        
        self.mocoSimUnd = MocoSimLayer(maxTrans=self.maxTrans, maxRot=self.maxRot, N_tp_Max=self.N_tp_Max, half_width_no_motion=self.half_width_no_motion, noMotionFrac=self.noMotionFrac)
        
        self.und = UndersampleLayer(center_fraction=self.center_fraction, acceleration=self.acceleration, center_lines=self.center_lines, variable_center=self.variable_center)

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.varnetmidc(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        kspace, _, target, _, _, max_value, _ = batch
                      
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        
        # --- varnet recon --- #    
        outksimgList, _ = self(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
        
        # loss_acc = []
        # for output in outksimgList:
            # target, output = transforms.center_crop_to_smallest(target, output)
            # loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
            # loss_acc.append(loss)
        # loss = sum(loss_acc)/len(outksimgList)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        kspace, _, target, fname, slice_num, max_value, _ = batch
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd.forward(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und.forward(kspace_sim)
        # --- varnet recon --- #   
        outksimgList, _ = self.forward(masked_kspace, mask)
        output = outksimgList[-1]
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        kspace, _, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet
        
        # --- simulate motion ---- #
        kspace_sim = self.mocoSimUnd(kspace)
        # ---- simulate undersampling --- #
        masked_kspace, mask = self.und(kspace_sim)
        # --- varnet recon --- #   
        outksimgList, _ = self(masked_kspace, mask)
        output = outksimgList[-1]

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        
        parser.add_argument(
            "--maxTrans",
            default=10.0,
            type=float,
            help="max translation in pixel for motion simulation",
        )
        
        parser.add_argument(
            "--maxRot",
            default=10.0,
            type=float,
            help="max rotation in degrees for motion simulation",
        )
        
        parser.add_argument(
            "--N_tp_Max",
            default=16,
            type=int,
            help="max number of motion events",
        )
        
        parser.add_argument(
            "--half_width_no_motion",
            default=12,
            type=int,
            help="half width around the center of k-space with no motion event",
        )
        
        parser.add_argument(
            "--noMotionFrac",
            default=0.25,
            type=float,
            help="Fraction of samples without any motion simulation",
        )
        
        parser.add_argument(
            "--center_fraction",
            default=0.08,
            type=float,
            help="Fully sampled Center fraction for undersampling",
        )
        
        parser.add_argument(
            "--variable_center",
            default=0.2,
            type=float,
            help="percentage of fully sampled center to vary",
        )
        
        parser.add_argument(
            "--center_lines",
            default=0,
            type=int,
            help="number of fixed center lines",
        )
        
        parser.add_argument(
            "--acceleration",
            default=4.0,
            type=float,
            help="Undersampling factor",
        )
        
        parser.add_argument(
            "--dc_chans",
            default=8,
            type=int,
            help="no of channels after first conv in midc model",
        )
        
        parser.add_argument(
            "--dc_pools",
            default=4,
            type=int,
            help="no of pools in midc model",
        )
        
        parser.add_argument(
            "--midc_w",
            default=320,
            type=int,
            help="input image widht for midc model",
        )
        
        parser.add_argument(
            "--midc_h",
            default=320,
            type=int,
            help="input image hight for midc model",
        )

        return parser



