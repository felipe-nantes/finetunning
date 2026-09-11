def test_python_is_311():
    import sys
    assert sys.version_info[:2] == (3, 11)
