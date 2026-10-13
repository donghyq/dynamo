# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


async def _stream():
    yield {"token_ids": [42]}


class _LegacyEngine:
    """Strict old signature: a leaked session argument raises TypeError."""

    def __init__(self):
        self.calls = []

    async def async_generate(
        self,
        input_ids,
        sampling_params,
        stream,
        external_trace_header,
        rid,
        data_parallel_rank,
        lora_path,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        image_data=None,
        audio_data=None,
        video_data=None,
    ):
        self.calls.append(
            {"input_ids": input_ids, "rid": rid, "bootstrap_room": bootstrap_room}
        )
        return _stream()


class _SessionEngine(_LegacyEngine):
    async def async_generate(self, session_id=None, require_reasoning=False, **kwargs):
        result = await super().async_generate(**kwargs)
        if session_id is not None:
            self.calls[-1]["session_id"] = session_id
        return result


async def _forward_stream(stream, *args, **kwargs):
    async for item in stream:
        yield item


async def _consume_stream(stream, context):
    async for _ in stream:
        pass


def _handler(mode, engine):
    cls = PrefillWorkerHandler if mode.startswith("prefill") else DecodeWorkerHandler
    handler = cls.__new__(cls)
    handler.engine = engine
    handler.enable_trace = False
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    if mode.startswith("prefill"):
        handler.bootstrap_host = "127.0.0.1"
        handler.bootstrap_port = 12345
        handler._consume_tasks = set()
        handler._consume_results = _consume_stream
    else:
        handler.serving_mode = (
            DisaggregationMode.DECODE
            if mode == "decode"
            else DisaggregationMode.AGGREGATED
        )
        handler.use_sglang_tokenizer = False
        handler._enable_frontend_decoding = False
        handler._mm_hashes_supported = False
        handler._routed_experts_kwargs = {}
        handler._build_sampling_params = lambda request: {"max_new_tokens": 1}
        handler._build_logprob_kwargs = lambda request: {}
        handler._metadata_uploader_from_request = lambda request: None
        handler._process_token_stream = _forward_stream
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prefill", "prefill_wrapped", "decode", "aggregated"])
@pytest.mark.parametrize("engine_cls", [_LegacyEngine, _SessionEngine])
@pytest.mark.parametrize("session_id", [None, "", "agent-session-1"])
async def test_generation_forwards_passive_session_identity(mode, engine_cls, session_id):
    engine = engine_cls()
    handler = _handler(mode, engine)
    request = {
        "token_ids": [10, 11, 12],
        "routing": {"dp_rank": 0},
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 12345,
            "bootstrap_room": 101,
        },
    }
    if session_id is not None:
        request["agent_context"] = {
            "session_id": session_id,
            "parent_session_id": "parent-session-1",
            "session_final": True,
        }
    if mode == "prefill_wrapped":
        request = {"request": request, "sampling_params": {"max_new_tokens": 1}}
    context = SimpleNamespace(id=lambda: "request-1", trace_id="trace-1")

    output = [item async for item in handler.generate(request, context)]

    assert output
    assert len(engine.calls) == 1
    assert engine.calls[0]["input_ids"] == [10, 11, 12]
    if mode.startswith("prefill") or mode == "decode":
        assert engine.calls[0]["bootstrap_room"] == 101
    if engine_cls is _SessionEngine and session_id:
        assert engine.calls[0]["session_id"] == session_id
    else:
        assert "session_id" not in engine.calls[0]
