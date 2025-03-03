import whois
import tldextract
from transformers import pipeline

def extract_domains_from_json(data):
    """
    Scan the JSON object for any string values that look like URLs or domains.
    Then, using tldextract, extract the base domain (domain.suffix).

    Args:
        data: A dictionary containing JSON data.

    Returns:
        A list of unique base domains extracted from the JSON.
    """
    domains = set()
    for key, value in data.items():
        if isinstance(value, str):
            # Check if the value contains potential URL/domain characters.
            if "http" in value or "." in value:
                ext = tldextract.extract(value)
                if ext.domain and ext.suffix:
                    base_domain = f"{ext.domain}.{ext.suffix}"
                    domains.add(base_domain)
    return list(domains)

def lookup_whois(domain):
    """
    Use the whois library to get registrant information for the given domain.

    Args:
        domain: The domain name to query.

    Returns:
        WHOIS information as a string, or None if an error occurs.
    """
    try:
        info = whois.whois(domain)
        return str(info)
    except Exception as e:
        print(f"Error looking up {domain}: {e}")
        return None

def build_combined_context(data, whois_data):
    """
    Build a combined context string from the JSON data and WHOIS responses.

    Args:
        data: The original JSON data.
        whois_data: A dictionary mapping domains to their WHOIS information.

    Returns:
        A single string that concatenates the JSON data and WHOIS info.
    """
    context_parts = [f"{key}: {value}" for key, value in data.items()]
    for domain, response in whois_data.items():
        context_parts.append(f"WHOIS data for {domain}: {response}")
    return " ".join(context_parts)

def answer_domain_ownership(context):
    """
    Use a Hugging Face question-answering pipeline to answer who owns the domain.

    Args:
        context: The combined context containing JSON and WHOIS data.

    Returns:
        The answer provided by the QA pipeline as a dictionary.
    """
    qa_pipe = pipeline(
        "question-answering",
        model="distilbert/distilbert-base-cased-distilled-squad",
        device_map=None,
        trust_remote_code=True,
    )
    question = "Which organization owns this domain?"
    result = qa_pipe(question=question, context=context)
    return result

def main():
    # Sample JSON input
    data = {
        "id": "org:github/azure",
        "alias": "Azure",
        "name": "azure",
        "repoUrl": "https://github.com/azure/azure-cli.git",
        "source": "github",
        "type": "O",
        "websiteUrl": "https://azure.com"
    }

    # Extract base domains from the JSON input
    base_domains = extract_domains_from_json(data)
    
    # If the JSON ID indicates a GitHub organization, exclude "github.com"
    if data.get("id", "").startswith("org:github"):
        base_domains = [d for d in base_domains if d.lower() != "github.com"]
    
    print("Extracted Domains:", base_domains)

    # Perform a WHOIS lookup for each extracted domain
    whois_responses = {}
    for domain in base_domains:
        whois_info = lookup_whois(domain)
        whois_responses[domain] = whois_info if whois_info else "No WHOIS info available"

    # Build the combined context from the JSON data and WHOIS responses
    combined_context = build_combined_context(data, whois_responses)
    
    # Use the question-answering pipeline to infer domain ownership
    qa_result = answer_domain_ownership(combined_context)
    
    print("\nQuestion-Answering Result:")
    print(qa_result)

if __name__ == "__main__":
    main()