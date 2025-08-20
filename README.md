This repo contains multiple models of GAN, trained to generate full body anime characters (mainly womans/girls), or you can train it on different dataset

**Install**
-
Download and install anaconda from: https://www.anaconda.com/download/success
and create env using these commands:
```
conda create --name mygan python=3.11 # This creates conda env with python3.11 (you can change mygan to whatever you want)

conda activate mygan # activates env

git clone https://github.com/28bc23/GAN-anime.git # clone repo to your working directory

cd GAN-anime # go to the main folder

pip3 install -r requirements.txt # install all needed requirements
```

If you want to just generate images just cpopy pretrained pth files in the same dir like is the main file GAN.py.
ex. for GAN.py:
```
cp pre-trainedModels/small/GAN/5160steps/* .
```

For training you need to download dataset below* and extract its content to folder called data, so it's same like in the picture below.
If you need to use different dataset(forexsample you don't want to generate anime characters), you need to prepair it, so it's the same way like the prefered dataset.
meaning in data folder you will need to have folders called exsactly 00XX and in these folders you need images in .png called 000XXX.png (X represents int value from 0 to 9).
or rewrite my code, so it fits to your dataset.

*Dataset: https://huggingface.co/datasets/skytnt/fbanimehq

<img width="397" height="490" alt="Example of folder layout for generating images from a pre-trained model and training your own model" src="https://github.com/user-attachments/assets/06d8a40e-814f-4be7-94c5-fd32155398b8" />

**Models**
-
- *smallGAN* - resolution 128x64 - most pre-trained model: 5160 steps
- *smallWGAN* - 128x64 **REMOVED**
- *PROGAN* - TODO: pre-train && optim, resolution 1024x512



**Generated Pictures**
-

**Small GAN** - this only exsample, It can be better if you train It more

<img width="64" height="128" alt="gen11" src="https://github.com/user-attachments/assets/e739cd49-fea8-4f23-853b-e9fb7229fb6b" />
<img width="64" height="128" alt="gen9" src="https://github.com/user-attachments/assets/2e0f14ed-ed07-4fcf-9ab4-f4d4f5505108" />
<img width="64" height="128" alt="gen8" src="https://github.com/user-attachments/assets/df5e26be-347d-4bd8-841b-2dc97d1a98a1" />
<img width="64" height="128" alt="gen7" src="https://github.com/user-attachments/assets/f167ff05-c046-4e01-86ae-e0ddcf04da7f" />
<img width="64" height="128" alt="gen6" src="https://github.com/user-attachments/assets/4e4beb86-0921-40f7-b10d-c56a0ed8bb7c" />
<img width="64" height="128" alt="gen5" src="https://github.com/user-attachments/assets/18258b49-f4c0-4c4c-9fac-9e440dd8ff4b" />
<img width="64" height="128" alt="gen400" src="https://github.com/user-attachments/assets/b69e86ba-3066-45e5-97a4-8172e5e42e27" />
<img width="64" height="128" alt="gen360" src="https://github.com/user-attachments/assets/5c620e0b-3dce-40c7-a669-c9564dded36f" />
<img width="64" height="128" alt="gen220" src="https://github.com/user-attachments/assets/fea6f5b5-30e0-48ea-ba58-e0bcd83abe01" />
<img width="64" height="128" alt="gen200" src="https://github.com/user-attachments/assets/2615f2cd-10ce-4ee7-a634-b07ace2276db" />
<img width="64" height="128" alt="gen190" src="https://github.com/user-attachments/assets/5e415bac-f1fc-4e36-aced-aa2d9432828e" />
<img width="64" height="128" alt="gen180" src="https://github.com/user-attachments/assets/0433b384-a595-4b65-b2bd-dd3f456c77fe" />
<img width="64" height="128" alt="gen170" src="https://github.com/user-attachments/assets/7a32c08c-ab05-4ef4-ba78-45747ffd9962" />
<img width="64" height="128" alt="gen160" src="https://github.com/user-attachments/assets/bc542b99-6355-4e9b-ad6b-1214fb34d8a2" />
<img width="64" height="128" alt="gen150" src="https://github.com/user-attachments/assets/90ddca5c-acd9-45a8-8f45-e3e5dfe4771b" />
<img width="64" height="128" alt="gen140" src="https://github.com/user-attachments/assets/c32a18a0-c118-4033-9cfe-0d08df5bebfd" />
<img width="64" height="128" alt="gen130" src="https://github.com/user-attachments/assets/ebd4ca36-b539-4509-87ad-0aac3561c7a7" />
<img width="64" height="128" alt="gen120" src="https://github.com/user-attachments/assets/7e63fe71-eda2-4964-a566-c1c188337309" />
<img width="64" height="128" alt="gen70" src="https://github.com/user-attachments/assets/63d6a1b4-6b9e-4601-b9fa-8798c62dd310" />
<img width="64" height="128" alt="gen19" src="https://github.com/user-attachments/assets/41db57d3-7eb2-440d-a1a8-16815bc26872" />
<img width="64" height="128" alt="gen18" src="https://github.com/user-attachments/assets/aadda9fd-5242-409a-a069-a6c78b97bdef" />
<img width="64" height="128" alt="gen17" src="https://github.com/user-attachments/assets/116d2bd5-da8f-4b80-854d-e9aeeffdfdf6" />
<img width="64" height="128" alt="gen15" src="https://github.com/user-attachments/assets/4fd48cb4-5746-47fd-96e8-a5f30fd28e21" />
<img width="64" height="128" alt="gen13" src="https://github.com/user-attachments/assets/411413e7-4cab-4adb-8533-26b8e055fa6b" />
<img width="64" height="128" alt="gen12" src="https://github.com/user-attachments/assets/17a03e08-462e-46c5-b21a-a84f46cd1cd3" />
<img width="64" height="128" alt="gen20" src="https://github.com/user-attachments/assets/e07f53b4-8576-4b19-970c-3b77906afa64" />
<img width="64" height="128" alt="gen3770" src="https://github.com/user-attachments/assets/27a9d5ef-5416-4d4c-a54d-8d0561669f4f" />
<img width="64" height="128" alt="gen3800" src="https://github.com/user-attachments/assets/0c8925f0-516f-43ab-8e32-d7d131fa1b89" />
<img width="64" height="128" alt="gen3740" src="https://github.com/user-attachments/assets/6b5026a7-830e-4651-b535-eae3bacf4792" />
<img width="64" height="128" alt="gen2940" src="https://github.com/user-attachments/assets/15da5259-4f0e-4c43-b4ea-5047cb9fc108" />

*I would appreciate feedback on how to improve my GANs, thx.*
