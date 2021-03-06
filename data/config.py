# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[12, 16],[24, 32],[48, 64],[96, 128],[192, 256]],
    'steps': [8, 16, 32,64,128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 1,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16],[32],[64],[128],[256]],
    'steps': [8, 16, 32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 1,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1},
    'in_channel': 256,
    'out_channel': 256
}

