import os
import sys
import subprocess
import tempfile
import time
from io import StringIO, BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from graphs import nx_to_adj_list

EXECUTABLE = os.path.join(os.path.dirname(__file__), "multi_genus_128")


def setup_multi_genus(adj_list):
    file = BytesIO()

    num_vertices = len(adj_list)
    min_vertex_id = min(min(neighbors) for neighbors in adj_list)
    num_bytes = 1 if num_vertices < 256 else 2
    file.write(num_vertices.to_bytes(num_bytes, byteorder="little"))
    for v in range(num_vertices):
        for u in adj_list[v]:
            u = u - min_vertex_id
            if u > v:
                file.write((u + 1).to_bytes(num_bytes, byteorder="little"))
        if v < num_vertices - 1:
            file.write((0).to_bytes(num_bytes, byteorder="little"))
    file.flush()

    return os.environ.copy(), file, "STDIN"


def run_multigenus(adj_list: list[list[int]]):
    with tempfile.NamedTemporaryFile(mode="w") as f:
        env, adj_file, inp_type = setup_multi_genus(adj_list)
        cmd = [EXECUTABLE]

        if inp_type == "FILE":
            assert isinstance(adj_file, StringIO)
            adj_file.seek(0)
            f.write(adj_file.read())  # type: ignore
            f.flush()

        start_time = time.perf_counter()
        with subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if inp_type == "STDIN" else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        ) as proc:
            if inp_type == "STDIN":
                assert isinstance(adj_file, BytesIO)
                assert proc.stdin is not None
                adj_file.seek(0)
                proc.stdin.write(adj_file.read())
                proc.stdin.close()

            assert proc.stderr is not None
            first_line = next(proc.stderr).decode("utf-8")
            if first_line.startswith("graphs with genus "):
                genus = int(first_line.split(":")[0].split(" ")[3])
                yield genus, "GENUS"
            yield first_line, "STDERR"
            for line in proc.stderr:
                yield line.decode("utf-8"), "STDERR"

            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.decode("utf-8"), "STDOUT"
        end_time = time.perf_counter()
        yield f"{end_time - start_time} seconds", "TIME"


def calc_genus(G) -> int:
    return next(run_multigenus(nx_to_adj_list(G)))[0]  # type: ignore
