import numpy as np

class EpsGreedy:
    def __init__(self, action_space_n, eps_start, eps_end, eps_eval, annealing_steps, warmup_steps):
        self.action_space_n = action_space_n
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_eval = eps_eval
        self.annealing_steps = annealing_steps
        self.warmup_steps = warmup_steps
        self.linear_decay = np.linspace(start=self.eps_start, stop=self.eps_end, num=self.annealing_steps)

    def select_action(self, q_values, step, train=True):
        eps = self._compute_current_eps(step) if train else self.eps_eval
        if np.random.rand() < eps:
            action = np.random.choice(self.action_space_n, size=1)[0]
        else:
            action = np.argmax(q_values)
        return action
    
    def _compute_current_eps(self, step):
        if step < self.warmup_steps:
            return self.eps_start
        elif step < self.annealing_steps + self.warmup_steps:
            return self.linear_decay[step - self.warmup_steps]
        else:
            return self.eps_end