from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayLR(_LRScheduler):
    """
    自定义学习率调度器，基于多项式衰减策略，支持 warmup 和 tot_updates 调控。
    """

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        """
        初始化多项式衰减学习率调度器。

        参数:
        - optimizer (Optimizer): 训练过程中要调整学习率的优化器。
        - warmup_updates (int): Warmup 阶段的更新次数，在此期间学习率线性增长。
        - tot_updates (int): 总更新次数，决定衰减阶段的结束点。
        - lr (float): 初始学习率（在 warmup 结束时达到的最大学习率）。
        - end_lr (float): 最终学习率（在 tot_updates 时的学习率）。
        - power (float): 多项式幂次，控制学习率衰减的曲线形状。
        - last_epoch (int): 上一个训练轮次的编号（默认值为 -1，表示从头开始）。
        - verbose (bool): 是否输出调度器的相关信息（默认为 False）。
        """
        self.warmup_updates = warmup_updates  # warmup 阶段的更新次数
        self.tot_updates = tot_updates  # 总的训练更新次数
        self.lr = lr  # warmup 阶段后达到的最大学习率
        self.end_lr = end_lr  # 最终衰减到的学习率
        self.power = power  # 多项式衰减的幂次
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)  # 调用父类构造函数

    def get_lr(self):
        """
        根据当前训练步数 (self._step_count) 动态计算每个参数组的学习率。

        返回:
        - list: 优化器中每个参数组的学习率列表。
        """
        if self._step_count <= self.warmup_updates:
            # Warmup 阶段：学习率线性增长
            self.warmup_factor = self._step_count / float(self.warmup_updates)  # 计算增长因子
            lr = self.warmup_factor * self.lr  # 当前学习率线性增长到 lr
        elif self._step_count >= self.tot_updates:
            # 衰减完成阶段：直接设置为最终学习率
            lr = self.end_lr
        else:
            # 多项式衰减阶段
            warmup = self.warmup_updates  # warmup 阶段的更新次数
            lr_range = self.lr - self.end_lr  # 学习率下降的总范围
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot_updates - warmup)  # 剩余的训练比例
            lr = lr_range * pct_remaining ** self.power + self.end_lr  # 多项式衰减公式

        # 为优化器中的每个参数组生成学习率
        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """
        此方法未实现，仅用于说明可扩展性。
        """
        assert False
