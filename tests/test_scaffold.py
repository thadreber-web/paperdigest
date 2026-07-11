def test_package_importable():
    import paperdigest

    assert paperdigest.__version__ == "0.1.0"
