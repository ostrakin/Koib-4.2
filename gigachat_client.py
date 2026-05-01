# -*- coding: utf-8 -*-
"""
KOIB RAG - GigaChat Client (сохранён из оригинального репозитория)
===================================================================
Клиент для работы с GigaChat API (Сбер).
Выполняет OAuth2 аутентификацию и отправку запросов к LLM.
"""

import logging
import time
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# Константы GigaChat API
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0.3
GIGACHAT_MAX_TOKENS = 1024
GIGACHAT_TIMEOUT = 30


class GigaChatClient:
    """
    Клиент для работы с GigaChat API.

    Атрибуты:
        credentials: Базовые учётные данные (client_id:client_secret в base64)
        access_token: Текущий access token
        token_expires_at: Время истечения токена
    """

    def __init__(self, credentials: str):
        self.credentials = credentials
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0

    def _get_token(self) -> str:
        """Получить или обновить access token через OAuth2."""
        if self.access_token and time.time() < self.token_expires_at - 60:
            return self.access_token

        logger.info("Получение нового токена GigaChat...")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {self.credentials}",
            "RqUID": "00000000-0000-0000-0000-000000000000"
        }

        payload = {"scope": GIGACHAT_SCOPE}

        try:
            response = requests.post(
                GIGACHAT_TOKEN_URL,
                headers=headers,
                data=payload,
                timeout=10,
                verify=False
            )

            if response.status_code != 200:
                raise RuntimeError(f"GigaChat OAuth error: {response.status_code}")

            data = response.json()
            self.access_token = data.get("access_token")
            expires_in = data.get("expires_in", 1800)
            self.token_expires_at = time.time() + expires_in

            return self.access_token

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"GigaChat connection error: {e}")

    def chat(self, prompt: str, temperature: float = GIGACHAT_TEMPERATURE,
             max_tokens: int = GIGACHAT_MAX_TOKENS) -> str:
        """Отправить запрос к GigaChat и получить ответ."""
        token = self._get_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        payload = {
            "model": GIGACHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                GIGACHAT_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=GIGACHAT_TIMEOUT
            )

            if response.status_code == 401:
                self.access_token = None
                token = self._get_token()
                headers["Authorization"] = f"Bearer {token}"
                response = requests.post(
                    GIGACHAT_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=GIGACHAT_TIMEOUT
                )

            if response.status_code != 200:
                return f"Ошибка GigaChat API: {response.status_code}"

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return "Не удалось получить ответ от сервиса."

            return choices[0].get("message", {}).get("content", "")

        except requests.exceptions.Timeout:
            return "Сервис временно недоступен (таймаут)."
        except Exception as e:
            return f"Ошибка: {e}"


def call_gigachat(prompt: str, credentials: str,
                  temperature: float = GIGACHAT_TEMPERATURE,
                  max_tokens: int = GIGACHAT_MAX_TOKENS) -> str:
    """Удобная функция для вызова GigaChat."""
    client = GigaChatClient(credentials)
    return client.chat(prompt, temperature, max_tokens)
