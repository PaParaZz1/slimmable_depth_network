from enum import Enum
import math
import random
import torch
import torch.nn as nn


class SlimBlockType(Enum):
    FAT = 1
    JOINT = 2
    MUSCLE = 3
    @classmethod
    def type2bn(cls, type):
        map_dict = {
            cls.FAT: "no_bn",
            cls.JOINT: "run_bn",
            cls.MUSCLE: "post_bn"
        }
        return map_dict[type]

    @classmethod
    def value2name(cls, value):
        if value not in cls.__members__.values():
            raise ValueError
        else:
            return cls.FAT


class SlimBNType(Enum):
    NORMAL_BN = 1
    RUN_NORM_BN = 2
    NO_BN = 3
    @classmethod
    def value2name(cls, value):
        if value not in cls.__members__.values():
            raise ValueError
        else:
            for k, v in cls.__members__.items():
                if v == value:
                    return k


class SlimBlock(object):
    def __init__(self, op, block_type):
        self.op = op
        self.block_type = block_type

    def __repr__(self):
        return '{}'.format(self.op)


class SlimBN2d(nn.Module):
    def __init__(self, *args, bn_type=SlimBNType.NORMAL_BN, **kwargs):
        super(SlimBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(*args, **kwargs)
        self.bn_type = bn_type
        self.bn_func_dict = {
            SlimBNType.NORMAL_BN: self._normal_bn,
            SlimBNType.NO_BN: self._no_bn,
            SlimBNType.RUN_NORM_BN: self._run_norm_bn
        }
        self.bn_func = self.bn_func_dict[bn_type]

    def _normal_bn(self, x):
        return self.bn(x)

    def _no_bn(self, x):
        return x

    def _run_norm_bn(self, x):
        epsilon = 1e-8
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        return (x - mean) / (std + epsilon)

    def forward(self, x):
        return self.bn_func(x)

    def __repr__(self):
        return 'SlimBN2d({}, bn_type={})'.format(self.bn, SlimBNType.value2name(self.bn_type))


class DepthSlimTrainStrategy(object):
    def __init__(self, necessary, option_num, option_block_num):
        N = int(math.pow(2, option_block_num))
        if necessary == 'sandwich':
            self.necessary = {0, N-1}
        elif necessary == 'max':
            self.necessary = {N-1}
        elif necessary == 'min':
            self.necessary = {0}
        else:
            raise ValueError
        self.option_block_num = option_block_num

        assert(option_num <= N - len(self.necessary))
        self.option_set = {x for x in range(N)} - self.necessary
        print('option_set', self.option_set)
        self.option_num = option_num

    def _get_binary_code_tuple(self, code):
        binary_code = bin(code)[2:]
        binary_code = binary_code.zfill(self.option_block_num)
        code = [int(x) for x in list(binary_code)]
        code = tuple(code)
        return code

    def get_necessary(self):
        result = []
        for item in self.necessary:
            result.append(self._get_binary_code_tuple(item))
        return result

    def _option_step(self):
        self.option = random.sample(self.option_set, self.option_num)

    def get_option(self):
        self._option_step()
        result = []
        for item in self.option:
            result.append(self._get_binary_code_tuple(item))
        return result


class DepthSlimStage(nn.Module):
    def __init__(self, blocks, train_strategy):
        super(DepthSlimStage, self).__init__()

        self.joints = []
        self.fats = nn.ModuleList()
        self.muscles = []
        for item in blocks:
            if item.block_type == SlimBlockType.FAT:
                self.fats.append(item.op)
            elif item.block_type == SlimBlockType.JOINT:
                self.joints.append(item.op)
            elif item.block_type == SlimBlockType.MUSCLE:
                self.muscles.append(item.op)
        self.train_strategy = train_strategy

        assert(len(self.joints) == 2)
        self.begin = self.joints[0]
        self.end = nn.Sequential(*self.muscles, self.joints[1])

    def forward(self, x):
        x = self.begin(x)
        x = self.end(x)
        return x

    def multi_transform(self, x):
        def binary_code_exe(func, code, x):
            print(code)
            assert(len(func) == len(code))
            for i in range(len(code)):
                if i == 0:
                    x = x
                else:
                    x = func[i](x)
            return x

        transform_x = []
        necessary = self.train_strategy.get_necessary()
        option = self.train_strategy.get_option()
        execute = necessary + option
        for item in execute:
            transform_x.append(binary_code_exe(self.fats, item, x))
        return transform_x

    def postrior_bn(self):
        for m in self.modules():
            if isinstance(m, SlimBN2d):
                if m.bn_type == SlimBNType.RUN_NORM_BN:
                    m.bn_type = SlimBNType.NORMAL_BN
                    continue
            m.requires_grad = False
            m.eval()

    def train_forward(self, x):
        x = self.begin(x)
        transform_x = self.multi_transform(x)
        result_x = []
        for x in transform_x:
            result_x.append(self.end(x))

        return result_x

    def __repr__(self):
        return ('DepthSlimStage' + '(\n' +
               'joint_begin: {}\n'.format(str(self.joints[0])) +
               'fat: (\n{}\n)\n'.format('\n'.join([str(x) for x in self.fats])) +
               'muscle: (\n{}\n)\n'.format('\n'.join([str(x) for x in self.muscles])) +
               'joint_end: {}\n'.format(str(self.joints[1])) +
               ')')