import subprocess

prompt = "hey can you like complain about some nonsense.\n"

process = subprocess.Popen(
    ["bash", "-c", "ollama run llama3.1:8b"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(prompt)

print(stdout)
