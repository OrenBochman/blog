import json
import fitz  # PyMuPDF

# Load the JSON annotations file
annotations_file = "annotations.json"
pdf_file = "paper.pdf"

with open(annotations_file, "r") as f:
    annotations_data = json.load(f)

annotations = annotations_data.get("annotations", [])

# Open the PDF document
doc = fitz.open(pdf_file)

# Debugging flag
debug = True

# Summary of annotation successes and failures
success_count = 0
failure_count = 0

for annotation in annotations:
    try:
        # Extract selectors
        target = annotation["target"][0]
        selectors = {selector["type"]: selector for selector in target["selector"]}
        page_index = selectors["PageSelector"]["index"]
        text_position = selectors.get("TextPositionSelector")
        text_quote = selectors.get("TextQuoteSelector")
        annotation_text = annotation.get("text", "")

        # Debug print for selector extraction
        if debug:
            print(f"Processing annotation ID: {annotation['id']}")
            print(f"Page index: {page_index}")
            print(f"Text position: {text_position}")
            if text_quote:
                print(f"Text quote: {text_quote}")
                prefix = text_quote.get("prefix", "")
                exact = text_quote.get("exact", "")
                suffix = text_quote.get("suffix", "")
                print(f"Prefix: {prefix}")
                print(f"Exact: {exact}")
                print(f"Suffix: {suffix}")

        # Access the appropriate page
        page = doc.load_page(page_index)

        if text_position:
            # Extract the start and end positions from the TextPositionSelector
            start = text_position["start"]
            end = text_position["end"]

            # Extract the text from the page to determine the exact range to highlight
            text = page.get_text("text")
            if len(text) >= end:
                # Extract the exact text based on the given character positions
                highlight_text = text[start:end]

                # Search for the extracted text to create a highlight
                instances = page.search_for(highlight_text)
                if instances:
                    # Highlight the correct instance
                    inst = instances[0]
                    highlight = page.add_highlight_annot(inst)
                    # Add a note to the highlight so it displays text when clicked
                    highlight.set_popup(fitz.Point(inst.x1, inst.y1))
                    highlight.set_info(info={"title": "Annotation", "content": annotation_text})
                    highlight.update()
                    success_count += 1
                else:
                    if debug:
                        print(f"Failed to match text position on page {page_index} for annotation ID: {annotation['id']}")
                    failure_count += 1
            else:
                if debug:
                    print(f"Text position out of range on page {page_index} for annotation ID: {annotation['id']}")
                failure_count += 1
        else:
            if debug:
                print(f"No valid TextPositionSelector found for annotation ID: {annotation['id']}")
            failure_count += 1

    except Exception as e:
        if debug:
            print(f"Error processing annotation ID: {annotation['id']} (Page: {page_index}): {e}")
        failure_count += 1

# Save the updated PDF
doc.save("highlighted_paper.pdf")
doc.close()

# Print summary
print(f"Annotations successfully matched and highlighted: {success_count}")
print(f"Annotations failed to match: {failure_count}")
