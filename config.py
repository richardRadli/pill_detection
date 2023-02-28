import argparse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T R A I N I N G +++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTraining:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
        self.parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16,
                                 help='Batch size')
        self.parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                                 help='Learning rate', dest='lr')
        self.parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
        self.parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
        self.parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                                 help='Percent of the data that is used as validation (0-100)')
        self.parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', '-ch', type=int, default=3, help="Number of channels")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T E S T I N G ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTesting:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        # Testing
        self.parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                                 help='Specify the file in which the model is stored')
        self.parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images',
                                 required=True)
        self.parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
        self.parser.add_argument('--viz', '-v', action='store_true',
                                 help='Visualize the images as they are processed')
        self.parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
        self.parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                                 help='Minimum probability value to consider a mask pixel white')
        self.parser.add_argument('--scale', '-s', type=float, default=0.5,
                                 help='Scale factor for the input images')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', '-ch', type=int, default=3, help="Number of channels")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   A U G M E N T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigAugment:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        # Testing
        self.parser.add_argument("--augmentation_factor", type=int, default=1)
        self.parser.add_argument("--use_original", type=bool, default=False)
        self.parser.add_argument("--use_random_rotation", type=bool, default=True)
        self.parser.add_argument("--rotation_angle", type=int, default=30)

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

        self.parser.add_argument("--type_of_network", type=str, default="Texture", help="RGB | Contour | Texture")
        self.parser.add_argument("--margin", type=float, default=1.0)
        self.parser.add_argument("--epochs", type=int, default=200)
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--learning_rate", type=float, default=2e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument("--save", type=bool, default=True)
        self.parser.add_argument("--save_freq", type=int, default=5)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
