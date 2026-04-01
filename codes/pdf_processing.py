import json
import pypdf
import ollama


def read_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def generate_report(text: str, report_type: str) -> str:
    prompts = {
        "analysis": "Sort states data by population and identify trends in growth over the past decade.",
        "summary": "Write a concise executive summary of the following document.",
        "key_points": "List the key points and findings from the following document as bullet points.",
        "entities": "Extract all people, organizations, locations, and dates from the following document. Return as JSON.",
    }
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": f"{prompts[report_type]}\n\nDocument:\n{text}"}],
        format="json" if report_type == "entities" else None,
    )
    return response.message.content


if __name__ == "__main__":
    text = read_pdf("tab87.pdf")

    reports = {
        "analysis": generate_report(text, "analysis"),
        "summary": generate_report(text, "summary"),
        "key_points": generate_report(text, "key_points"),
        "entities": json.loads(generate_report(text, "entities")),
    }

    for name, content in reports.items():
        print(f"\n{'='*40}\n{name.upper()}\n{'='*40}")
        print(json.dumps(content, indent=2) if isinstance(content, dict) else content)
