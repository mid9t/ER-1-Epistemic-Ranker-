import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end / network-heavy tests")
