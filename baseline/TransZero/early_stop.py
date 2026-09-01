from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np
from torch.nn import Module


class StopVariable(Enum):
    """
    定义早停的变量类型，枚举类。
    - LOSS: 监控损失值，用于判断早停。
    - ACCURACY: 监控准确率，用于判断早停。
    - NONE: 不启用早停。
    """
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()


class Best(Enum):
    """
    定义模型状态保存策略，枚举类。
    - RANKED: 按照排名保存模型状态，针对单一指标变化。
    - ALL: 同时保存满足所有指标条件时的模型状态。
    """
    RANKED = auto()
    ALL = auto()


def Stop_args(patience=100, max_epochs=2000):
    """
    生成早停参数的默认值。

    参数:
    - patience (int): 容忍的最大无改进步数。
    - max_epochs (int): 最大训练轮数。

    返回:
    - dict: 包含默认的早停参数。
    """
    return dict(
        stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS],  # 默认同时监控准确率和损失值
        patience=patience,
        max_epochs=max_epochs,
        remember=Best.RANKED  # 保存按排名的模型状态
    )


class EarlyStopping:
    """
    早停类，用于监控训练过程中指标的变化，并根据预设条件决定是否停止训练。
    """

    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        """
        初始化 EarlyStopping 对象。

        参数:
        - model (Module): 被监控的模型（用于保存其状态）。
        - stop_varnames (List[StopVariable]): 要监控的指标列表，如损失值或准确率。
        - patience (int): 最大无改进步数。
        - max_epochs (int): 最大训练轮数。
        - remember (Best): 保存模型状态的策略（RANKED 或 ALL）。
        """
        self.model = model  # 被监控的模型
        self.comp_ops = []  # 比较操作符列表（如 <= 或 >=）
        self.stop_vars = []  # 监控的指标名称（如 'loss' 或 'acc'）
        self.best_vals = []  # 每个监控指标的最佳值

        # 初始化监控变量和比较操作
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)  # 损失值：越小越好，用 <= 比较
                self.best_vals.append(np.inf)  # 损失值的初始最佳值为正无穷
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)  # 准确率：越大越好，用 >= 比较
                self.best_vals.append(-np.inf)  # 准确率的初始最佳值为负无穷

        self.remember = remember  # 决定如何保存模型状态
        self.remembered_vals = copy.copy(self.best_vals)  # 保存的最佳值（可回溯）
        self.max_patience = patience  # 最大容忍的无改进步数
        self.patience = self.max_patience  # 当前剩余的容忍步数
        self.max_epochs = max_epochs  # 最大训练轮数
        self.best_epoch = None  # 最佳模型所在的 epoch
        self.best_state = None  # 保存的最佳模型状态（字典形式）

    def check(self, values: List[np.floating], epoch: int) -> bool:
        """
        检查当前指标值是否触发早停条件。

        参数:
        - values (List[np.floating]): 当前的指标值列表（与 stop_vars 对应）。
        - epoch (int): 当前训练轮数。

        返回:
        - bool: 是否需要早停（True 表示需要早停）。
        """
        # 检查每个指标是否比当前的最佳值更好
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]

        if any(checks):  # 如果有任何一个指标改善
            # 更新最佳值
            self.best_vals = np.choose(checks, [self.best_vals, values])
            # 重置 patience
            self.patience = self.max_patience

            # 检查是否需要保存当前模型状态
            comp_remembered = [
                self.comp_ops[i](val, self.remembered_vals[i])
                for i, val in enumerate(values)
            ]
            if self.remember is Best.ALL:
                # ALL 模式：只有当所有指标都改善时才保存
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    # 保存模型状态
                    self.best_state = {
                        key: value.cpu() for key, value
                        in self.model.state_dict().items()
                    }
            elif self.remember is Best.RANKED:
                # RANKED 模式：逐一比较，每个指标若改善即保存
                for i, comp in enumerate(comp_remembered):
                    if comp:  # 如果某个指标有改善
                        if not (self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            # 保存模型状态
                            self.best_state = {
                                key: value.cpu() for key, value
                                in self.model.state_dict().items()
                            }
                            break
                    else:
                        break
        else:
            # 如果没有改善，则减少 patience
            self.patience -= 1

        # 当 patience 减少到 0 时，触发早停
        return self.patience == 0

    def simple_check(self, loss_list):
        """
        简单的早停检查函数，仅基于损失值判断。

        参数:
        - loss_list (list): 最近几个 epoch 的损失值。

        返回:
        - int: 是否需要早停（1 表示需要早停，0 表示不需要）。
        """
        if len(loss_list) <= self.patience:
            return 0  # 如果损失列表长度不足 patience，无需早停
        else:
            flag = 1  # 假设需要早停
            for i in range(self.patience):
                # 如果当前损失值小于前几个 epoch 的损失值，则不早停
                if loss_list[-1] < loss_list[-1 - i]:
                    flag = 0
            return flag
