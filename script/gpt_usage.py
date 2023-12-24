# the async version is adapted from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a

import os
import openai
import ast
import asyncio
from typing import List

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
class OpenAIChat_Embed:
    # support chat and embedding
    def __init__(self, chat_model_name='gpt-3.5-turbo', max_tokens=4000, temperature=0.7, top_p=1, request_timeout=240,
                 embed_model_name='text-embedding-ada-002'):
        if "gpt-3.5" in chat_model_name:
            max_tokens = 4000
        elif "gpt-4" in chat_model_name:
            max_tokens = 8000
        else:
            pass
        self.config = {
            'chat_model_name': chat_model_name,
            'embed_model_name': embed_model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,
        }
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # openai.api_base = os.getenv("OPENAI_API_BASE")

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
                    if mode=='chat':
                        response = await openai.ChatCompletion.acreate(
                            model=self.config['chat_model_name'],
                            messages=messages,
                            # max_tokens=self.config['max_tokens'],
                            temperature=self.config['temperature'],
                            top_p=self.config['top_p'],
                            request_timeout=self.config['request_timeout'],
                        )
                        # print(response)
                        return response['choices'][0]['message']['content']
                    elif mode=='embed':
                        response = await openai.Embedding.acreate(input=messages, model=self.config['embed_model_name'])
                        return response["data"][0]["embedding"]
                    else:
                        raise NotImplementedError
                except openai.error.RateLimitError:
                    print('Rate limit error, waiting for 4 second...')
                    await asyncio.sleep(30)
                except openai.error.APIConnectionError:
                    print('API Connection error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.APIError:
                    print('API error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.Timeout:
                    print('Timeout error, waiting for 1 second...')
                    await asyncio.sleep(30)
                except openai.error.ServiceUnavailableError:
                    print('Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)



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
    chat = OpenAIChat_Embed(chat_model_name='gpt-4-1106-preview')

    predictions = asyncio.run(chat.async_run(
        messages_list=[
                          [
                              {
                                  "role": "user",
                                  "content": "show either 'ab' or '['a']'. Do not do anything else."
                              }
                          ],
                      ]*1,
        mode="chat"
    ))

    for pred in predictions:
        print(pred)
