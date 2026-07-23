import json
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from dual_search.protocol import get_tool_schemas


# Pinned fixture of the tool-related branches in Qwen3-VL's Apache-2.0 chat
# template. The omitted branches only handle image/video placeholder expansion.
# Source: Qwen/Qwen3-VL-4B-Instruct, chat_template.json, revision d51ae92.
QWEN3_VL_TOOL_TEMPLATE = r"""
{%- if tools %}<|im_start|>system
# Tools
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools><|im_end|>
{%- endif %}
{%- for message in messages %}
{%- if message.role == "user" %}<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" %}<|im_start|>assistant
{{ message.content }}
{%- for tool_call in message.tool_calls or [] %}
{%- set call = tool_call.function if tool_call.function else tool_call %}
<tool_call>
{"name": "{{ call.name }}", "arguments": {{ call.arguments if call.arguments is string else call.arguments | tojson }}}
</tool_call>
{%- endfor %}<|im_end|>
{%- elif message.role == "tool" %}<|im_start|>user
<tool_response>
{{ message.content }}
</tool_response><|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{%- endif %}
"""


def _tokenizer():
    backend = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = QWEN3_VL_TOOL_TEMPLATE
    return tokenizer


class QwenChatTemplateTest(unittest.TestCase):
    def test_schema_native_calls_and_tool_responses_render_together(self):
        tokenizer = _tokenizer()
        messages = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "<think>inspect</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "vision_search",
                            "arguments": {"image_index": 1, "query": "striped wings"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "vision_search", "content": "Caption 1 evidence"},
            {
                "role": "assistant",
                "content": "<think>lookup</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "species habitat"}, separators=(",", ":")),
                        },
                    }
                ],
            },
            {"role": "tool", "name": "search", "content": "Doc 1 evidence"},
            {"role": "assistant", "content": "<think>done</think><answer>forest</answer>"},
        ]

        rendered = tokenizer.apply_chat_template(
            messages,
            tools=get_tool_schemas(),
            tokenize=False,
            add_generation_prompt=False,
        )

        self.assertEqual(rendered.count("<tools>"), 1)
        self.assertEqual(rendered.count("<tool_call>"), 2)
        self.assertEqual(rendered.count("<tool_response>"), 2)
        self.assertIn('"additionalProperties": false', rendered)
        self.assertIn('"image_index": 1', rendered)
        self.assertIn('"query":"species habitat"', rendered)
        self.assertIn("<answer>forest</answer>", rendered)


if __name__ == "__main__":
    unittest.main()
