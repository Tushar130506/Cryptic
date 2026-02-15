import config


def test_validate_config():
    assert config.validate_config() is True
