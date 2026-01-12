import subprocess

prompt = "Russia stationed troops along the border with Ukraine, is this likely to escalate or de-escalate? give me a likelihood from low, mid or high\n"

process = subprocess.Popen(
    ["bash", "-c", "ollama run llama3.1:8b"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(prompt)

print(stdout)
