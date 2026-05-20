from datasets import load_dataset


def main():
    dataset = load_dataset(
        "WINGNUS/ACL-OCL",
        split="test",
        streaming=True,
    )

    for i, item in enumerate(dataset.take(3)):
        print("=" * 80)
        print("Example", i)
        print("Keys:", list(item.keys()))

        for k, v in item.items():
            print(f"\n[{k}]")
            text = str(v)
            print(text[:1500])

    print("Done.")


if __name__ == "__main__":
    main()