# Space image-to-image translation with Pix2Pix and Aladin Sky Atlas

This project uses the conditional GAN Pix2Pix in combination with the Aladin Sky Atlas software to generate images of a portion of the sky in a desired wavelength from images of the same portion of the sky in other wavelengths. 
The code of the original Pix2Pix to input a dataset has been slightly modified in order to accept several input images, thus allowing the model to go not only from B to A but from B + C + D… to A. 
To use this project just follow the instructions in the original [Pix2Pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix9) but using the modified code and multiinput dataset here.

## Motivation and theoretical rationale
Different sky surveys have yielded data about portions of the sky in different wavelengths, with different resolutions and with a different total coverage. Very few of them are all-sky surveys, most of the data available has either very limited coverage (HST) or low resolution (2MASS). This results in parts of the sky where there might be data available in different resolutions and wavelengths but not in others. This project intends to find out if it is possible and feasible to use that available and complementary information to generate a target image in the unavailable wavelength/resolution.

From a theoretical point of view, the physical law most relevant to our use case is Planck’s law. This law describes the spectral density of electromagnetic radiation emitted by a black body in thermal equilibrium at a given temperature T. A black-body is an idealized object which absorbs and emits all radiation frequencies. Planck radiation has a maximum intensity at a wavelength that depends on the temperature of the body ([Wikipedia](https://en.wikipedia.org/wiki/Planck%27s_law)).


  <img align="center" src="https://user-images.githubusercontent.com/108660081/182445316-5c914dd2-76a6-4630-b442-be5027ea1635.png" width="400"/>  
  <img align="center" src="https://user-images.githubusercontent.com/108660081/182445881-8093107d-358f-4b29-8723-749766a62127.png" width="400"/>
                  <sub>Family of curves for different temperatures (left) and the Sun approximated as a black-body (right)</sub>

In other words, Planck’s law shows that it may be possible to use complementary wavelengths (astronomical observations) to reconstruct a missing one.

## Dataset creation
Each data point or element in the dataset should look like this:

![imagen](https://user-images.githubusercontent.com/108660081/182447267-f2db0b77-10cc-44e3-ae1e-a7490eb39d4a.png)

A single image made up of several (4 in this case) horizontally concatenated images of the same object from different observations/wavelengths (SDSS9 (high-res optical), GALEX (UV), 2MASS (IR) and DSS2 (low-res optical), respectively in the previous example).

To create a dataset for training and/or a data point for inference the desired images have to be extracted from Aladin, a sky atlas software. The script aladin_data.py contains a basic example to automatically extract the 7840 astronomical objects in the New General Catalogue (NGC) in four different wavelengths from Aladin and concatenate them into a dataset. It is possible to get any other object or portion of the sky by modifying the script. The name of astronomical objects as well as specific coordinates can be used as input to extract images from Aladin.

## Pix2Pix code modification
For the model to correctly accept data points like the example above and process them into several input images and one output image, the last part of the code aligned_dataset.py was modified as follows. This code assumes that 4 images of 3 channels each are fed, crops them, transforms them and concatenates the inputs as tensors (3 inputs and 1 output, B+C+D =>A) (in principle it should be possible to allow an arbitrary number of input images with a more efficient modification):

```
[...]
import torch
[...]
        # split AB image into A and B (concatenated B+C+D)
        w, h = AB.size
        w4 = int(w / 4)
        A = AB.crop((0, 0, w4, h))
        B = AB.crop((w4, 0, w4*2, h))
        C = AB.crop((w4*2, 0, w4*3, h))
        D = AB.crop((w4*3, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = B_transform(C)
        D = B_transform(D)
        B = torch.cat((B, C, D))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

```

Additionally, the current original Pix2Pix code does not support visualization of images with a number of channels different from 1 or 3. To avoid an error, the function called tensor2im in util.py has to be modified by adding the following:

```
[...]
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] == 9:  
            image_numpy, b, c = np.vsplit(image_numpy, 3)
[...]
```

This will just resplit the input images that were concatenated to feed the network and return one of them for visualization as real B (origin) for reference.

## Some results and comments
A Pix2Pix model was trained on a dataset of 3019 images of objects of the NGC to go from low quality optical (DSS2) to better quality optical (SDSS9), using (i) only low quality optical, (ii) low quality optical + IR and (iii) low quality optical + IR + UV. The training was stopped at around 90 epochs given a non-improving divergence in the adversarial loss and the model at epoch 65 (before divergence started) used for testing. The results were compared in the W&B platform using the generator loss as metric and by visual inspection.

The loss function of the generator in Pix2Pix is made up of two terms: the adversarial loss and the L1 loss:

![imagen](https://user-images.githubusercontent.com/108660081/182449312-2547c60c-4e81-4dc8-8e80-8a17a527c702.png)

According to Jason Brownlee from [Machine Learning Mastery](https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix-generative-adversarial-network/) “*The adversarial loss influences whether the generator model can output images that are plausible in the target domain, whereas the L1 loss regularizes the generator model to output images that are a plausible translation of the source image*”. 

The parameter importance tool from W&B shows that there is a relevant negative correlation between the adversarial loss and the parameter input_nc (number of input channels, i.e., number of input images), that is, the higher the number of input images the lower the loss, as expected. Regarding the L1 loss, the correlation varied during training between positive and negative while comparing only the three abovementioned runs (low quality optical vs low quality optical + IR vs low quality optical + IR + UV) but was negative if comparing all other previous experimental runs (~7 smaller runs with less epochs, images, etc.). My best guess about this is that either the extra information is “confusing” the generator somehow or outliers are distorting the correlation, according to W&B docs “*correlations are sensitive to outliers, which might turn a strong relationship to a moderate one, specially if the sample size of hyperparameters tried is small*”.

Although a visual comparison of the results is hard and subjective given the high similarity of the images, there seems to be several repeated patterns that could point to some conclusions about using additional information from several wavelengths (red circle = incorrect, green circle = correct):

It seems to reinforce generation confidence when there are overlapping astronomical objects:

![imagen](https://user-images.githubusercontent.com/108660081/182451643-05a34365-7d3f-4211-b95b-29cff990a35f.png)

Better rendering of brightness and color of point-like objects:
![imagen](https://user-images.githubusercontent.com/108660081/182451832-3ed2c0fa-c55b-487e-a716-c7e3e6f691f6.png)

Incorrect inclusion of extremely faint groups of small background point-like objects (possible confusion due to additional information):
![imagen](https://user-images.githubusercontent.com/108660081/182452201-8546bab0-8351-49da-9f5b-dd1f0e68014d.png)

With regard to this last issue, many instances were detected where the additional information seemed to have actually helped “turn off” those faint background point-like objects in optical +IR+UV while being shown (incorrectly) if using only optical as source image.

Further experimentation with more complex images and additional wavelengths is needed but, judging by the results, the model seems to be using the additional information successfully to constrain and better render the target image. Specially interesting is the fact that color is better translated when using information from additional wavelengths, this is in line with Plank’s black-body radiation, visual appearance and radiation spectral density (and temperature) are interrelated.

## Acknowledgments 
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

This project has made use of "Aladin sky atlas" developed at CDS, Strasbourg Observatory, France 
→ [2000A&AS..143...33B](http://cdsads.u-strasbg.fr/cgi-bin/nph-bib_query?2000A%26AS..143...33B&db_key=AST&nosetcookie=1) and [2014ASPC..485..277B](http://cdsads.u-strasbg.fr/cgi-bin/nph-bib_query?2014ASPC..485..277B&db_key=AST&nosetcookie=1).

