import warnings
class DefaultConfig(object):
    env="default"
    model='ResNet34'
    train_data_root='./data/train/'
    test_data_root='./data/test1'
    load_model_path=None#'D:/workspace/code/pytorch/DogCat/models/checkpoints/AlexNet_3.pth'
    batch_size=64
    use_gpu = False
    num_workers=0
    print_freq=20

    debug_file='./debug'
    result_file='./result.csv'

    max_epoch=100
    lr=0.1
    lr_decay=0.95
    weight_decay=1e-4

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning :opt has not attribut %s %k')
            setattr(self, k, v)
            # 打印配置信息
            print('user config:')
            for k, v in self.__class__.__dict__.items():
                if not k.startwith('__'):
                    print(k, getattr(self, k))
