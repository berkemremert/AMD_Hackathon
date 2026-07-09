# The judging VM runs linux/amd64. If you build on Apple Silicon (M1/M2/M3), build with:
#   docker buildx build --platform linux/amd64 -t <your-image> --push .
# On Windows/Intel/AMD a normal build already produces linux/amd64.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project in.
COPY . .

# The harness mounts /input (with tasks.json) and /output at run time, then runs this.
# main.py reads /input/tasks.json and writes /output/results.json, then exits 0.
CMD ["python", "src/main.py"]
