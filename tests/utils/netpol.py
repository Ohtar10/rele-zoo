from relezoo.networks.simple import SimpleFC


def build_net(in_shape: int, out_shape: int):
    """Builds a simple network as per in/out shape."""
    return SimpleFC(in_shape, out_shape)

