# import argparse
from argparse import Namespace


def set_config():
    config = Namespace(
        in_channels=3,
        channels=64,
        num_layer=8,
        image_path=r'00000_01_00.png',
        sr_factor=2,
        num_workers=4,
        patch_size=128,
        batch_size=8,
        num_scale=6,
        exp_name='output_pics_00000_01_00',
        # num_epoch=40,
        num_epoch=8000,
        lr=1e-3,
        # check_val_every_n_epoch=10,
        check_val_every_n_epoch=100,
        accelerator='gpu'
    )
    
    return config

