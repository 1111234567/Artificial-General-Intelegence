import gzip
import json
import math
from experiments.wait_content import eval as e
from experiments.wait_content.eval_test import source


def test_real_unstarted_ready_time_serializes_without_world_mutation(tmp_path):
    world, h = source()
    assert bool(world.ready_time.isinf().any())
    before = e.state_digest(world.state_dict())
    path = tmp_path/'actual-world.json.gz'
    e.compressed_json(path, dict(ready_time=world.ready_time.tolist(), tick=world.tick,
                               active=world.active.tolist(), finite=1.25))
    saved = json.loads(gzip.decompress(path.read_bytes()))
    assert saved['finite'] == 1.25 and saved['tick'] == world.tick
    assert saved['active'] == world.active.tolist()
    assert all(value is None for row in saved['ready_time'] for value in row)
    assert before == e.state_digest(world.state_dict())
