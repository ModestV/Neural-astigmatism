"""Переносимость адаптеров: загрузка, переключение и комбинирование LoRA-адаптеров (этап 4).

Идея: базовая модель грузится один раз, роли (программист, юрист, репетитор)
переключаются через PeftModel.load_adapter / set_adapter без перезагрузки весов.
"""


class AdapterManager:
    def __init__(self, base_model):
        self.base_model = base_model
        self.loaded: dict[str, str] = {}  # имя адаптера -> путь

    def load(self, name: str, path: str) -> None:
        raise NotImplementedError("Этап 4")

    def switch(self, name: str) -> None:
        raise NotImplementedError("Этап 4")

    def combine(self, names: list[str], weights: list[float], new_name: str) -> None:
        """Взвешенная комбинация адаптеров (peft add_weighted_adapter)."""
        raise NotImplementedError("Этап 4")
