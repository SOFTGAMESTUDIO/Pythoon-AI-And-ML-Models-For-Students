def search_keyword(pages, keyword):
    results = []

    for i, text in enumerate(pages, start=1):
        lines = text.split("\n")

        for line in lines:
            if keyword.lower() in line.lower():
                results.append({
                    "page": i,
                    "line": line.strip()
                })

    return results