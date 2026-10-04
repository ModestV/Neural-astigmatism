"""Локальный UI: чат с базовой моделью и переключение ролей через LoRA-адаптеры (этап 5).

Запуск: python app/app.py
"""

import gradio as gr

from peft_lab.registry import list_adapters


def respond(message: str, history: list, adapter: str) -> str:
    # TODO(этап 5): AdapterManager.switch(adapter) и генерация ответа локальной моделью
    return f"[{adapter}] Модель ещё не подключена. Вы написали: {message}"


def build_app() -> gr.ChatInterface:
    adapters = list_adapters()
    return gr.ChatInterface(
        fn=respond,
        title="peft-lab — локальный персональный ассистент",
        additional_inputs=[
            gr.Dropdown(
                choices=adapters, value=adapters[0] if adapters else None, label="Роль (адаптер)"
            )
        ],
    )


if __name__ == "__main__":
    build_app().launch()
