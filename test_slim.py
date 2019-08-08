import slim
import torch
import torch.nn as nn


def conv2d(bn_type):
    return nn.Sequential(
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(inplace=True),
        slim.SlimBN2d(64, bn_type=bn_type)
    )


def get_blocks():
    model = [
        slim.SlimBlock(conv2d(slim.SlimBNType.NORMAL_BN), slim.SlimBlockType.JOINT),
        slim.SlimBlock(conv2d(slim.SlimBNType.NO_BN), slim.SlimBlockType.FAT),
        slim.SlimBlock(conv2d(slim.SlimBNType.NO_BN), slim.SlimBlockType.FAT),
        slim.SlimBlock(conv2d(slim.SlimBNType.RUN_NORM_BN), slim.SlimBlockType.MUSCLE),
        slim.SlimBlock(conv2d(slim.SlimBNType.RUN_NORM_BN), slim.SlimBlockType.MUSCLE),
        slim.SlimBlock(conv2d(slim.SlimBNType.NORMAL_BN), slim.SlimBlockType.JOINT),
    ]
    return model


def test():
    blocks = get_blocks()
    strategy = slim.DepthSlimTrainStrategy(necessary='sandwich', option_num=1, option_block_num=2)
    stage = slim.DepthSlimStage(blocks, strategy)
    # stage structure
    #print(stage)
    # forward
    inputs = torch.randn(2, 64, 8, 8)
    output = stage(inputs)
    print(output.shape)
    # forward cuda
    stage = stage.cuda()
    inputs = inputs.cuda()
    output = stage(inputs)
    print('forward cuda', output.shape)
    # strategy train
    for i in range(10):
        output_list = stage.train_forward(inputs)
        for item in output_list:
            print(item.shape)
        print('strategy train over num{}'.format(i))
    # postrior_bn
    stage.postrior_bn()
    output = stage(inputs)
    print(output.shape)


if __name__ == "__main__":
    test()
