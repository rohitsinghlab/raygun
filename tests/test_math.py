# tests/test_math.py

import subprocess
import random


def test_addition():
    assert 1 + 1 == 2
    

# raygun-embed -t test.fasta --device cpu
def test_embed_cli_runs(tmpdir):
    random.seed(123)
    aa_seq = ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=53))
    fasta = tmpdir.join("test.fasta")
    fasta.write(f">test_protein\n{aa_seq}")

    try:
        result = subprocess.run(
            ["raygun-embed", "-t", str(fasta), "--device", "cpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True  # raises CalledProcessError on non-zero exit
        )
    except subprocess.CalledProcessError as e:
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise  # Re-raise to fail the test
