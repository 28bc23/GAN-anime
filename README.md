*I would appreciate feedback on how to improve this GAN, thx.*

This is GAN trained for generating anime girls in resolution 128x64

**Dataset I used**: https://huggingface.co/datasets/skytnt/fbanimehq/blob/main/data/fbanimehq-00.zip


**Install**
For installation you need to clone this repo and install dependencies from requirements.txt, it's good choise to use venv like conda
If you want to just generate images just extract pretrained pth files in the same dir like is the main file GAN.py and file starting with *g* rename to *generator.pth* and same with file starting with *d* rename to *discriminator.pth*.

For training you need to download dataset above and extract its content to folder called data, so it's same like in the picture below.
If you need to use different dataset(forexsample you don't want to generate anime girls), you need to prepair it, so it's the same way like the prefered dataset.
meaning in data folder you will need to have folders called exsactly 000X and in these folders you need images in .png called 000XXX.png (X represents int value from 0 to 9).
or rewrite my code, so it fits to your dataset

**Models**
- *smallGAN* - resolution 128x64
- *mediumGAN* - in progress
- *largeGAN* - in progress, resolution 1024x512

**Tested resolutions**
  - *128x64*
  - *1024x512* - *This resolution is too big so youu will be getting collored noise, you need to modity the code so it's working propertly on higher resolutions*

<img width="397" height="490" alt="Example of folder layout for generating images from a pre-trained model and training your own model" src="https://github.com/user-attachments/assets/06d8a40e-814f-4be7-94c5-fd32155398b8" />


**Small GAN**
- 980 epochs: 