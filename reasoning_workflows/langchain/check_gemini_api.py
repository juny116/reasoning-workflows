"""Send one Gemini request to check the configured API key."""
import argparse
import os


def main():
    from langchain_google_genai import GoogleGenerativeAI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-pro")
    parser.add_argument("--prompt", default="Reply with a short greeting.")
    args = parser.parse_args()
    llm = GoogleGenerativeAI(model=args.model, google_api_key=os.environ["GOOGLE_API_KEY"])
    print(llm(args.prompt))


if __name__ == "__main__":
    main()
