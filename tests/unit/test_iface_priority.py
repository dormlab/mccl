import os


def test_set_iface_priority_sets_env():
    import mccl
    mccl.set_iface_priority(["192.168.103.", "192.168.102.", "192.168."])
    assert os.environ["MCCL_IFACE_PRIORITY"] == "192.168.103.,192.168.102.,192.168."


def test_default_priority_applied_on_import():
    import mccl  # noqa: F401
    val = os.environ.get("MCCL_IFACE_PRIORITY", "")
    assert val, "default MCCL_IFACE_PRIORITY should be set after import"
    assert "192.168." in val
