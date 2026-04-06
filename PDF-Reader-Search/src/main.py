from reader import read_pdf
from formatter import format_text
from search import search_keyword

def main():
    path = input("Enter PDF path: ")
    pages = read_pdf(path)

    while True:
        print("\n1. Read PDF")
        print("2. Search")
        print("3. Exit")

        choice = input("Choose: ")

        if choice == "1":
            for i, page in enumerate(pages, start=1):
                print("\n" + "="*60)
                print(f"PAGE {i}")
                print("="*60 + "\n")
                print(format_text(page))

        elif choice == "2":
            keyword = input("Enter keyword: ")
            results = search_keyword(pages, keyword)

            if not results:
                print("No results found.")
            else:
                for r in results:
                    print(f"[Page {r['page']}] {r['line']}")

        elif choice == "3":
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()