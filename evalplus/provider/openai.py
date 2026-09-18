import os
from typing import List

import openai

from evalplus.gen.util import openai_request
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import concurrent_call


class OpenAIChatDecoder(DecoderBase):
    def __init__(
        self, name: str, base_url=None, enable_thinking: bool = False, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.enable_thinking = enable_thinking

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)
        prompt = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        # use concurrency based batching for o1 and deepseek models
        if self.name.startswith("o1-") or self.name == "deepseek-chat":
            return self._codegen_batch_via_concurrency(prompt, num_samples)

        return self._codegen_api_batch(prompt, batch_size)

    def _codegen_api_batch(self, prompt: str, batch_size: int) -> List[str]:
        client = openai.OpenAI(
            # `or "none"`: a set-but-empty key yields an illegal `Bearer ` header.
            api_key=os.getenv("OPENAI_API_KEY") or "none", base_url=self.base_url
        )
        if self.enable_thinking:
            kwargs = {"enable_thinking": self.enable_thinking}
        else:
            kwargs = {}
        ret = openai_request.make_auto_request(
            client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            **kwargs,
        )

        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)

        return outputs

    def _codegen_batch_via_concurrency(self, prompt: str, batch_size: int) -> List[str]:
        batches = concurrent_call(
            batch_size, self._codegen_api_batch, prompt, batch_size=1
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False

    async def async_codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)
        prompt = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        return await self._async_codegen_api_batch(prompt, batch_size)

    async def _async_codegen_api_batch(self, prompt: str, batch_size: int) -> List[str]:
        client = openai.AsyncOpenAI(
            # `or "none"`: a set-but-empty key yields an illegal `Bearer ` header.
            api_key=os.getenv("OPENAI_API_KEY") or "none",
            base_url=self.base_url,
            timeout=1200,
        )
        if self.enable_thinking:
            kwargs = {"enable_thinking": self.enable_thinking}
        else:
            kwargs = {}

        ret = await openai_request.async_make_auto_request(
            client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            **kwargs,
        )

        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)
        return outputs
