from abc import ABC


class Model(ABC):
    def __init__(self, model_name: str):
        self.model = None
        self.name = model_name

    def size(self):
        """
            Compute the size in MB of the model
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

    def summary(self):
        """
            Number of total parameters and number of trainable parameters
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model Summary:\n- Total Parameters: {total_params:,}\n- Trainable Parameters: {trainable_params:,}"

    def model_name(self) -> str:
        """
            :return the name of the model
        """
        return self.name

    def get_model(self):
        """
            :return the pytorch model
        """
        return self.model

    def set_model(self, model):
        """
            Set the model
        """
        self.model = model

    def quantize_model(self) -> None:
        """
            Quantize the model in fp16 (Floating Point 16 bits)
        """
        self.model.half()
