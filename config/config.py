import argparse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T R A I N I N G +++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTrainingUnet:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
        self.parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8)
        self.parser.add_argument('--momentum', type=float, default=0.999)
        self.parser.add_argument('--gradient_clipping', type=float, default=1.0)
        self.parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
        self.parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
        self.parser.add_argument('--valid', '-v', type=float, default=10.0,
                                 help='Percent of the data that is used as validation (0-100)')
        self.parser.add_argument('--amp', type=bool, default=False, help='Use mixed precision')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', '-ch', type=int, default=3, help="Number of channels")
        self.parser.add_argument('--save_checkpoint', '-sc', type=bool, default=True)
        self.parser.add_argument('--load_mask_values', type=bool, default=True)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T E S T I N G ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTestingUnet:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        # Testing
        self.parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                                 help='Specify the file in which the model is stored')
        self.parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images',
                                 required=True)
        self.parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
        self.parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
        self.parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                                 help='Minimum probability value to consider a mask pixel white')
        self.parser.add_argument('--scale', '-s', type=float, default=0.5,
                                 help='Scale factor for the input images')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', type=int, default=3)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   N E T W O R K +++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigStreamNetwork:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--type_of_net", type=str, default="EfficientNetSelfAttention",
                                 help="StreamNetwork | EfficientNet | EfficientNetSelfAttention")
        self.parser.add_argument("--type_of_stream", type=str, default="LBP", help="RGB | Contour | Texture | LBP")
        self.parser.add_argument("--train_rate", type=float, default=0.8)
        self.parser.add_argument("--margin", type=float, default=0.5)
        self.parser.add_argument("--epochs", type=int, default=30)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--learning_rate_cnn_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_cnn_con_tex", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_con_tex", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_ensa_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_ensa_con_tex", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument("--img_size", type=int, default=128)
        self.parser.add_argument("--load_ref_vector", type=bool, default=False)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   N E T W O R K +++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigFusionNetwork:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--type_of_net", type=str, default="StreamNetwork",
                                 help="StreamNetwork | EfficientNet | EfficientNetSelfAttention")
        self.parser.add_argument("--margin", type=float, default=0.5)
        self.parser.add_argument("--train_split", type=float, default=0.8)
        self.parser.add_argument("--epochs", type=int, default=30)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--learning_rate", type=float, default=1e-2)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument("--img_size", type=int, default=128)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
