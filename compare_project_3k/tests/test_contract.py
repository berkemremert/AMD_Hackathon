import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))


class TestOutputContract(unittest.TestCase):
    """The harness reads the output file it mounts, possibly as a different
    user, and matches results on the echoed task_id. Guard both."""

    def _run(self, tasks):
        with tempfile.TemporaryDirectory() as d:
            inp, outp = os.path.join(d, "in.json"), os.path.join(d, "out.json")
            with open(inp, "w") as fh:
                json.dump(tasks, fh)
            env = dict(os.environ, INPUT_PATH=inp, OUTPUT_PATH=outp,
                       FIREWORKS_API_KEY="x", FIREWORKS_BASE_URL="http://x",
                       ALLOWED_MODELS="dummy-4b")
            # No network: solve() will fail and _answer_one returns "" — we
            # only assert the contract (ids present, valid JSON, file mode).
            subprocess.run([sys.executable, "main.py"], cwd=ROOT, env=env,
                           capture_output=True, timeout=60)
            with open(outp) as fh:
                results = json.load(fh)
            mode = os.stat(outp).st_mode & 0o777
            return results, mode

    def test_ids_and_permissions(self):
        tasks = [{"task_id": 7, "prompt": "hi"},
                 {"task_id": "t2", "prompt": "yo"},
                 {"prompt": "no id"}]
        results, mode = self._run(tasks)
        self.assertEqual([r["task_id"] for r in results], [7, "t2", "idx_2"])
        self.assertTrue(all("answer" in r for r in results))
        self.assertEqual(mode & 0o044, 0o044, f"results file not world-readable: {oct(mode)}")


if __name__ == "__main__":
    unittest.main()
