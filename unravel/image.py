import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
from unravel.tabular import UnRAVELTabularExplainer
from torchvision import transforms


class UnRAVELImageExplainer(UnRAVELTabularExplainer):
    def __init__(self, bbox_model, mode="classification"):
        """[summary]

        Args:
            bbox_model ([type]): [description]
            mode (str, optional): [description]. Defaults to "classification".
            verbosity (bool, optional): [description]. Defaults to False.
        """
        self.bbox_model = bbox_model
        self.mode = mode

    def generate_domain(self, arr, interval=0):
        """[summary]

        Args:
            arr ([type]): [description]

        Returns:
            [type]: [description]
        """
        exploration_neighborhood = []
        for idx in range(arr.shape[0]):
            idx_domain = {
                "name": f"var_{idx}",
                "type": "discrete",
                "domain": [0, 1],
            }

            exploration_neighborhood.append(idx_domain)

        return exploration_neighborhood

    def bin2img(self, arr):
        """[summary]

        Args:
            arr ([type]): [description]

        Returns:
            [type]: [description]
        """

        img = copy.deepcopy(self.img_original)

        # Assuming arr to be 2-D
        zeros = np.where(arr.ravel() == 0)[0]
        mask = np.zeros(self.segments.shape).astype(bool)
        for z in zeros:
            mask[self.segments == z] = True
        img[mask] = self.img_perturbed[mask]

        return img

    def preprocess_img(self, img):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        return transf(img).unsqueeze(0)

    def f_p(self, arr, return_classidx=False):
        """[summary]

        Args:
            arr ([type]): [description]
            return_classidx (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        img = self.bin2img(arr)
        # Converting numpy image to PIL
        img_processed = self.preprocess_img(Image.fromarray(img.astype("uint8"), "RGB"))
        self.bbox_model.eval()
        logits = self.bbox_model(img_processed)
        probs = F.softmax(logits, dim=1)
        probs_top = probs.topk(1)

        if not return_classidx:
            print(
                self.img_original_arg, probs[0, self.img_original_arg].detach().numpy()
            )
        plt.imshow(img)
        plt.show()
        plt.close()
        return (
            [probs_top.values.detach().numpy(), probs_top.indices.detach().numpy()]
            if return_classidx
            else probs[0, self.img_original_arg].detach().numpy()
        )

    def generate_init(self, img):
        """Initializes segmentations, pertubed image, X_init and Y_init

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """

        segments = slic(
            np.array(img), n_segments=50, compactness=10, sigma=1, start_label=1
        )

        self.segments = segments
        self.no_segments = np.unique(segments).shape[0]

        self.img_original = img

        img_perturbed = img.copy()

        for x in np.unique(segments):
            img_perturbed[segments == x] = (
                np.mean(img[segments == x][:, 0]),
                np.mean(img[segments == x][:, 1]),
                np.mean(img[segments == x][:, 2]),
            )

        self.img_perturbed = img_perturbed

        # self.img_perturbed = np.zeros(img.shape)
        self.img_original_prob, self.img_original_arg = self.f_p(
            img, return_classidx=True
        )

        X_init = np.ones((1, self.no_segments))

        return X_init

    def explain(
        self,
        img,
        kernel_type="RBF",
        max_iter=50,
        alpha="EI",
        jitter=5,
        normalize=True,
        plot=False,
        interval=1,
        verbosity=True,
    ):
        # img is in numpy form

        X_init = self.generate_init(img)

        return super().explain(
            X_init=X_init,
            feature_names=[],
            kernel_type=kernel_type,
            max_iter=max_iter,
            alpha=alpha,
            jitter=jitter,
            normalize=normalize,
            plot=plot,
            interval=interval,
            verbosity=verbosity,
            maximize=True,
        )
