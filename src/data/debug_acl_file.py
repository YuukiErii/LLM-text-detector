import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "external_human"
    / "academic"
    / "acl_ocl"
    / "Base_JSON"
    / "prefixA"
    / "json"
    / "acl"
    / "2020.acl-main.100.json"
)


def main():
    print("File:", TARGET_FILE)
    print("Exists:", TARGET_FILE.exists())
    print("Size:", TARGET_FILE.stat().st_size if TARGET_FILE.exists() else "N/A")

    text = TARGET_FILE.read_text(encoding="utf-8", errors="ignore")
    print("\nFirst 1000 chars:")
    print(text[:1000])

    try:
        data = json.loads(text)
        print("\nJSON loaded successfully.")
        print("Type:", type(data))

        if isinstance(data, dict):
            print("Keys:", list(data.keys()))

            for key in ["paper_id", "title", "abstract", "pdf_parse"]:
                print(f"\n[{key}]")
                value = data.get(key)
                print(type(value))
                print(str(value)[:1500])

            pdf_parse = data.get("pdf_parse", {})
            if isinstance(pdf_parse, dict):
                body_text = pdf_parse.get("body_text", [])
                print("\nbody_text type:", type(body_text))
                print("body_text length:", len(body_text))

                if body_text:
                    print("\nFirst body paragraph:")
                    print(body_text[0])

        elif isinstance(data, list):
            print("List length:", len(data))
            if data:
                print("First item type:", type(data[0]))
                print("First item:", str(data[0])[:1500])

    except Exception as e:
        print("\nFailed to load JSON:")
        print(repr(e))


if __name__ == "__main__":
    main()