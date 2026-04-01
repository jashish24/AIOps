import json
import ollama

TEXT = """OpenAI was founded in December 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, Elon Musk and others.
The company is headquartered in San Francisco, California. Its mission is to ensure that artificial general
intelligence benefits all of humanity. OpenAI released ChatGPT in November 2022, which became one of the
fastest-growing consumer applications in history."""

PROMPT = f"""Extract structured details from the text below. Return valid JSON only with these keys:
- people: list of names
- organizations: list of org names
- locations: list of locations
- dates: list of dates or time references
- products: list of product names
- mission: mission/goal statement if any
- summary: one-sentence summary

Text: {TEXT}"""


def extract_details(text: str) -> dict:
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": PROMPT}],
        format="json",
    )
    return json.loads(response.message.content)


if __name__ == "__main__":
    result = extract_details(TEXT)
    print(json.dumps(result, indent=2))
