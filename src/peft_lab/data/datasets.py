"""Загрузка и подготовка датасетов в chat-формате (этап 2). Формат описан в data/README.md."""


def load_chat_dataset(path: str):
    """Читает JSONL с полем messages и возвращает datasets.Dataset."""
    raise NotImplementedError("Этап 2")


def apply_chat_template(dataset, tokenizer, max_seq_length: int):
    """Превращает диалоги в токены с помощью chat template модели."""
    raise NotImplementedError("Этап 2")
