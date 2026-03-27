import torch
import torch.nn as nn
import torch.nn.init as init

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, encoder_filters, decoder_filters):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = self.build_encoder(encoder_filters)
        self.decoder = self.build_decoder(decoder_filters)
        self._initialize_weights()

    def build_encoder(self, filters):
        layers = []
        in_channels = 3
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def build_decoder(self, filters):
        layers = []
        in_channels = filters[0]
        for out_channels in filters[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, x):
        return self.decoder(x)

# Ejemplo de uso
# input_image = torch.randn(1, 3, 256, 256)  # Entrada con 1 batch, 3 canales, 256x256

# encoder_filters = [64, 128, 256, 512, 1024, 2048]
# decoder_filters = [2048, 1024, 512, 256, 128, 64]

# model = ConvolutionalAutoencoder(encoder_filters, decoder_filters).to(device)

# output_image = model(input_image)
# print(f"Input size: {input_image.size()}")
# print(f"Output size: {output_image.size()}")


