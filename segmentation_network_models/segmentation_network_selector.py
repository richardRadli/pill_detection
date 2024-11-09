from abc import ABC, abstractmethod

from segmentation_network_models.w2_network import W2Net
import segmentation_models_pytorch as smp


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class W2NetWrapper(BaseModel):
    def __init__(self, cfg):
        self.model = (
            W2Net(
                in_channels=cfg.get("channels"),
                out_channels=cfg.get("classes")
            )
        )

    def forward(self, x):
        return self.model(x)


class UNetWrapper(BaseModel):
    def __init__(self, cfg):
        self.model = (
            smp.Unet(
                encoder_name=cfg.get("encoder_name"),
                encoder_weights=cfg.get("encoder_weights"),
                in_channels=cfg.get("channels"),
                classes=cfg.get("classes")
            )
        )

    def forward(self, x):
        return self.model(x)


class SegmentationNetworkFactory:
    _network_map = {
        "W2Net": W2NetWrapper,
        "UNet": UNetWrapper
    }

    @staticmethod
    def create_model(network_type, cfg):
        if network_type not in SegmentationNetworkFactory._network_map:
            raise ValueError(f"Network type {network_type} not supported")

        model_wrapper_class = SegmentationNetworkFactory._network_map[network_type]
        model = model_wrapper_class(cfg).model

        return model
