import os


def load_global_context(context_source):
    """
    Load global context from various sources:
    - A single file path (string).
    - A list of file paths.
    - A directory (all files with specific extensions).
    
    Returns a string that concatenates the contents of all files.
    """
    context_texts = []

    try:
        # If a list of file paths is provided.
        if isinstance(context_source, list):
            for file_path in context_source:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        context_texts.append(f.read())
                else:
                    print(f"File not found or is not a file: {file_path}")

        # If a directory is provided, read all relevant files in that directory.
        elif os.path.isdir(context_source):
            for filename in os.listdir(context_source):
                # You can change the extensions as needed.
                if filename.lower().endswith((".txt", ".csv", ".json")):
                    file_path = os.path.join(context_source, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        context_texts.append(f.read())

        # If it's a single file path.
        elif os.path.exists(context_source) and os.path.isfile(context_source):
            with open(context_source, "r", encoding="utf-8") as f:
                context_texts.append(f.read())

        else:
            print(f"Invalid context source: {context_source}")

        global_context = "\n\n".join(context_texts)
        print("global_context:", global_context)
        return global_context

    except Exception as e:
        print("global_context:", "")
        print(f"Error reading global context: {e}")
        return ""