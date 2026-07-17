from web.paper_manager import PaperManager, JobState


def test_status_idle_initially():
    m = PaperManager()
    st = m.status()
    assert st["active"] is False
    assert st["job"] is None


def test_read_outputs_missing_returns_empty():
    m = PaperManager()
    # 无输出文件时返回空结构，不抛异常
    assert m.equity() == []
    assert m.trades() == []
    assert m.metrics() == {}
