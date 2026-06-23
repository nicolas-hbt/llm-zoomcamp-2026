import time

from tqdm.auto import tqdm
from rag_helper import RAGBase

from dataclasses import dataclass


@dataclass
class CostResult:
    input_cost: float
    output_cost: float
    total_cost: float

def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    # Note: Use prompt_tokens and completion_tokens for modern OpenAI objects
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    
    return CostResult(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost
    )

def calc_total_price(usages):
    # Using a generator expression for cleaner code
    return sum(calc_price(usage).total_cost for usage in usages)


def llm_structured(client, instructions, user_prompt, output_type, model="openai/gpt-oss-120b"): # Use a compatible model
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output", # Must follow ^[a-zA-Z0-9_-]+$
                "schema": output_type.model_json_schema()
            }
        }
    )

    # Parse the content string back into Pydantic model
    content = response.choices[0].message.content
    parsed_output = output_type.model_validate_json(content)
    
    return parsed_output, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="openai/gpt-oss-120b", # Use a compatible model
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
