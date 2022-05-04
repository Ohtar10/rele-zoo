from relezoo.networks.simple import SimpleFC, SimpleFCMuSigma


def build_net(in_shape: int, out_shape: int, n_type: str = "simple"):
    """Builds a simple network as per in/out shape."""
    if n_type == "simple":
        return SimpleFC(in_shape, out_shape)
    else:
        return SimpleFCMuSigma(in_shape, out_shape)

