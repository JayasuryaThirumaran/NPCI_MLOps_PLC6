import os

# Define the required Dockerfile keywords (some are alternatives)
REQUIRED_KEYWORDS = {
    "FROM": False,
    "ADD_OR_COPY": False,
    "RUN": False,
    "CMD_OR_ENTRYPOINT": False,
}

def check_dockerfile(path="Dockerfile"):
    if not os.path.isfile(path):
        print(f"❌ Dockerfile not found at: {path}")
        return

    with open(path, "r") as file:
        lines = file.readlines()

    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith("FROM"):
            REQUIRED_KEYWORDS["FROM"] = True
        elif stripped.startswith("ADD") or stripped.startswith("COPY"):
            REQUIRED_KEYWORDS["ADD_OR_COPY"] = True
        elif stripped.startswith("RUN"):
            REQUIRED_KEYWORDS["RUN"] = True
        elif stripped.startswith("CMD") or stripped.startswith("ENTRYPOINT"):
            REQUIRED_KEYWORDS["CMD_OR_ENTRYPOINT"] = True

    print("🔍 Dockerfile Keyword Check:")
    for keyword, found in REQUIRED_KEYWORDS.items():
        status = "✅ Found" if found else "❌ Missing"
        print(f"{keyword}: {status}")

if __name__ == "__main__":
    check_dockerfile()
