class WarmupScheduler:

    def __init__(self, optimizer, model_size, factor, warmup_steps):
        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size

        self.cur_step = 0
        self._rate = 0

    def step(self):
        self.cur_step += 1
        step = self.cur_step

        rate = self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

        for p in self.optimizer.param_groups:
            p["lr"] = rate

        self._rate = rate

    def get_last_lr(self):
        return [self._rate]
