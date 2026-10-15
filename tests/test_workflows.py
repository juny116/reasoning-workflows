from reasoning_workflows.data import read_jsonl
from reasoning_workflows.configuration import configured


def test_shared_data_loader_reads_jsonl(tmp_path):
    path = tmp_path / 'input.jsonl'; path.write_text('{"question":"example"}\n')
    assert read_jsonl(path) == [{"question": "example"}]

def test_config_loader_uses_only_explicit_input(tmp_path, monkeypatch):
    path = tmp_path / 'execution.yaml'; path.write_text('answer: 42\n')
    monkeypatch.setattr('sys.argv', ['workflow', '--config', str(path)])
    @configured
    def execute(config): return config.answer
    assert execute() == 42
