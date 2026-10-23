"""Drosselt Anfragen an das Sprachmodell auf ein Limit pro Minute.

Manche API-Keys erlauben nur wenige Anfragen pro Minute (Projektlimit). Beim Streaming meldet OpenAI ein
überschrittenes Limit nicht als HTTP 429, sondern als Fehler im bereits geöffneten Stream; weder das SDK noch ein
HTTP-Transport können die Anfrage dann wiederholen. Deshalb wird vor dem Senden gewartet, statt das Limit zu reißen.
Das gilt je Prozess: Laufen App und Evaluation gleichzeitig, teilen sie sich das Limit trotzdem.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import ModelMessage, ModelResponse, RunContext
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

WINDOW_SECONDS = 60.0
# Abstand zur Fenstergrenze, damit Uhrabweichungen zum Server nicht doch zu einem 429 führen
MARGIN_SECONDS = 1.0


class RequestBudget:
    """Gleitendes Fenster: höchstens `per_minute` Anfragen in jeden 60 Sekunden.

    Ein Token-Bucket (Schub, danach eine Anfrage je 60/`per_minute` s) reichte nicht: OpenAI lehnte damit
    gelegentlich ab, zählt also mindestens so streng wie dieses Fenster.
    """

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self.sleep = sleep
        self.sent: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wartet, bis eine Anfrage erlaubt ist; gibt die Wartezeit in Sekunden zurück."""
        async with self._lock:
            waited = 0.0
            while True:
                now = self.clock()
                while self.sent and now - self.sent[0] >= WINDOW_SECONDS:
                    self.sent.popleft()
                if len(self.sent) < self.per_minute:
                    break
                wait = WINDOW_SECONDS - (now - self.sent[0]) + MARGIN_SECONDS
                await self.sleep(wait)
                waited += wait
            self.sent.append(self.clock())
            return waited


class RateLimitedModel(WrapperModel):
    """Modell, das jede Anfrage erst sendet, wenn das Budget es erlaubt."""

    def __init__(self, wrapped: Model | str, per_minute: int) -> None:
        super().__init__(wrapped)
        self.budget = RequestBudget(per_minute)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self.budget.acquire()
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        await self.budget.acquire()
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream
