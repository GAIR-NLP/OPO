# the async version is adapted from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a

import os
import openai
import ast
import asyncio
from typing import List
from anthropic import AsyncAnthropic
import anthropic
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
class Claude:
    # support chat and embedding
    def __init__(self, chat_model_name='claude-2', max_tokens_to_sample=1000,temperature=0.7, top_p=1, timeout=600):
        if "gpt-3.5" in chat_model_name:
            max_tokens = 4000
        elif "gpt-4" in chat_model_name:
            max_tokens = 8000
        else:
            pass
        self.config = {
            'chat_model_name': chat_model_name,
            'max_tokens_to_sample': max_tokens_to_sample,
            'temperature': temperature,
            'top_p': top_p,
            'timeout': timeout,
        }
        self.claude_api_key = "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"
        # openai.proxy = "http://127.0.0.1:7900"
        self.model = AsyncAnthropic(api_key=self.claude_api_key, proxies="http://127.0.0.1:7900")
        # openai.organization = "org-f8AZxuNv8B4NE2ByndrAknGE"
        openai.proxy = "http://127.0.0.1:7900"
        # openai.api_base = "http://openai.plms.ai/v1"

        # openai.api_base = "https://api.aigcbest.top/v1"


    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
            self,
            messages_list,
            mode
    ) -> List[str]:
        """Dispatches requests to OpenAI API asynchronously. support chat and embed

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(messages, retry=3):
            for _ in range(retry):
                try:

                    # here the message format is [{"role":"system","content":system_prompt},
                    # {"role":"user","content":user_prompt}]
                    message = messages[0]["content"]+'\n'+messages[1]["content"]
                    prompt = f"{AsyncAnthropic.HUMAN_PROMPT} {message} {AsyncAnthropic.AI_PROMPT}"
                    res = await self.model.completions.create(
                        model=self.config["chat_model_name"],
                        prompt=prompt,
                        max_tokens_to_sample=1000,
                        temperature=self.config["temperature"]
                    )
                    # print(prompt)
                    # print(res.completion)
                    return res.completion

                except anthropic.APIConnectionError as e:
                    print("The server could not be reached, restart the claude model")
                    self.model = AsyncAnthropic(api_key=self.claude_api_key, proxies="http://127.0.0.1:7900")
                except anthropic.RateLimitError as e:
                    print("A 429 status code (rate limit) was received; we should back off a bit.")
                    await asyncio.sleep(10)
                except anthropic.APIStatusError as e:
                    print("Another non-200-range status code was received")
                    print(e.status_code)
                    print(e.response)
                    await asyncio.sleep(10)
                    self.model = AsyncAnthropic(api_key=self.claude_api_key, proxies="http://127.0.0.1:7900")

            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list, mode='chat'):
        retry = 50
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
                mode=mode
            )

            preds = [prediction if prediction is not None else None for prediction
                     in predictions]

            finished_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finished_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finished_index]

            retry -= 1

        return responses


if __name__ == "__main__":
    chat = Claude(chat_model_name='claude-2')

    predictions = asyncio.run(chat.async_run(
        messages_list=[
                          [
                              {
                                  "role": "system",
                                  "content": "you are a helpful assistant."
                              },
                              {
                                  "role": "user",
                                  "content": "show either 'ab' or '['a']'. Do not do anything else."
                              }
                          ],
                      ]*10,
        mode="chat"
    ))

    for pred in predictions:
        print(pred)
