*I would appreciate feedback on how to improve this GAN, thx.*

This is GAN trained for generating anime girls in resolution 128x64

**Dataset I used**: https://huggingface.co/datasets/skytnt/fbanimehq/blob/main/data/fbanimehq-00.zip


**Install**

For installation you need to clone this repo and install dependencies from requirements.txt, it's good choise to use venv like conda
If you want to just generate images just cpopy pretrained pth files in the same dir like is the main file GAN.py.

For training you need to download dataset above and extract its content to folder called data, so it's same like in the picture below.
If you need to use different dataset(forexsample you don't want to generate anime girls), you need to prepair it, so it's the same way like the prefered dataset.
meaning in data folder you will need to have folders called exsactly 000X and in these folders you need images in .png called 000XXX.png (X represents int value from 0 to 9).
or rewrite my code, so it fits to your dataset

**Models**
- *smallGAN* - resolution 128x64
- *smallWGAN* - 128x64
- *PGGAN* - TODO, resolution 1024x512

<img width="397" height="490" alt="Example of folder layout for generating images from a pre-trained model and training your own model" src="https://github.com/user-attachments/assets/06d8a40e-814f-4be7-94c5-fd32155398b8" />


**Small GAN - 2800 epochs**

<img width="64" height="128" alt="gen0" src="https://github.com/user-attachments/assets/78b8ddc1-158f-4e1a-a675-f4fac3cc9e7d" />
<img width="64" height="128" alt="gen8" src="https://github.com/user-attachments/assets/0291490b-cec6-4dda-b243-3dd93d5754f9" />
<img width="64" height="128" alt="gen9" src="https://github.com/user-attachments/assets/3cd80df7-ba6c-4c01-a5ae-82aab2633024" />
<img width="64" height="128" alt="gen12" src="https://github.com/user-attachments/assets/ec7495a0-f05f-40d6-9c3a-9ad6bf519e0d" />
<img width="64" height="128" alt="gen13" src="https://github.com/user-attachments/assets/14a5890c-4f19-45b7-b488-1bd5b016ed46" />
<img width="64" height="128" alt="gen18" src="https://github.com/user-attachments/assets/22b34738-0093-495b-9b28-9313d805e9a9" />
<img width="64" height="128" alt="gen19" src="https://github.com/user-attachments/assets/dce76114-129b-4476-8b0a-908c1d7316d6" />
<img width="64" height="128" alt="gen11" src="https://github.com/user-attachments/assets/866b4749-a42f-4411-abe6-154a327277ec" />
<img width="64" height="128" alt="gen15" src="https://github.com/user-attachments/assets/1d0f4447-9957-42e5-9129-a86bd31a6d5b" />
<img width="64" height="128" alt="gen7" src="https://github.com/user-attachments/assets/c3dca1f6-214a-4a0e-b310-f7e34867d083" />
<img width="64" height="128" alt="gen6" src="https://github.com/user-attachments/assets/53d2aea5-e8a1-48fc-82f6-7596a0af8ae5" />
<img width="64" height="128" alt="gen5" src="https://github.com/user-attachments/assets/a62aadfb-1961-4633-971b-0479ea2293a1" />
<img width="64" height="128" alt="gen17" src="https://github.com/user-attachments/assets/ece8968e-53bf-4221-ba0a-06e691ba7cde" />
