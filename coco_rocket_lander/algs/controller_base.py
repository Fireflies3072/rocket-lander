from abc import ABC, abstractmethod


class ControllerBase(ABC):
    @abstractmethod
    def reset(self, initial_state):
        pass

    @abstractmethod
    def update(self, target, measurement, dt: float):
        pass