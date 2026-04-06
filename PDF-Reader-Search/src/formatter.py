def format_text(text):
    lines = text.split("\n")
    paragraph = ""
    formatted = []

    for line in lines:
        line = " ".join(line.split())

        if not line:
            continue

        if line.endswith((".", "!", "?")):
            paragraph += " " + line
            formatted.append(paragraph.strip())
            paragraph = ""
        else:
            paragraph += " " + line

    if paragraph:
        formatted.append(paragraph.strip())

    return "\n\n".join(formatted)