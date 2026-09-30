from saverouter.smoke import run_smoke_test


def test_end_to_end_smoke():
    result = run_smoke_test()
    assert result["status"] == "PASS"
    assert result["observations"] == 240
    assert result["density"] == 1 / 3
