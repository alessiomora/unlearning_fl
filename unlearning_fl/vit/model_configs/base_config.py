import ml_collections


# CONFIG_ViT_T: ConfigDict = {
#     "dropout": 0.1,
#     "mlp_dim": 768,
#     "num_heads": 3,
#     "num_layers": 12,
#     "hidden_size": 192,
# }
#
# CONFIG_ViT_S: ConfigDict = {
#         "dropout": 0.1,
#         "mlp_dim": 768,
#         "num_heads": 6,
#         "num_layers": 12,
#         "hidden_size": 384,
#     }

def get_config(
    model_name: str = "deit_tiny_patch16_224",
    resolution: int = 224,
    patch_size: int = 16,
    projection_dim: int = 192,
    num_layers: int = 12,
    num_heads: int = 3,
    init_values: float = None,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    pre_logits: bool = False,
) -> ml_collections.ConfigDict:
    """Default initialization refers to deit_tiny_patch16_224 for ImageNet-1k.

    Reference:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/deit.py#L141
    """
    config = ml_collections.ConfigDict()
    config.name = model_name

    config.input_shape = (resolution, resolution, 3)
    config.image_size = resolution
    config.patch_size = patch_size
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 1000

    config.initializer_range = 0.02
    config.layer_norm_eps = 1e-6
    config.num_layers = num_layers

    config.classifier = "token"
    config.init_values = init_values
    config.drop_path_rate = drop_path_rate

    config.pre_logits = pre_logits

    if model_name == "deit_tiny_patch16_224":
        config.dropout_rate = dropout_rate
        config.num_heads = num_heads
        config.projection_dim = 192
        config.mlp_units = [
            config.projection_dim * 4,
            config.projection_dim,
        ]

    elif model_name == "deit_small_patch16_224":
        config.dropout_rate = 0.1
        config.num_heads = 6
        config.projection_dim = 384
        config.mlp_units = [
            config.projection_dim * 4,
            config.projection_dim,
        ]

    else:  # model_name == "deit_base"
        config.dropout_rate = 0.1
        config.num_heads = 12
        config.projection_dim = 768
        config.mlp_units = [
            config.projection_dim * 4,
            config.projection_dim,
        ]

    return config.lock()
