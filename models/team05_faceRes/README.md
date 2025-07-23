# 02_faceRestoration

 <pre>
Requirements:
conda env create -f environment.yml
 </pre>

Download the pretrained models listed in `model_zoo/team05_faceRes/team02_faceRes.txt` and place them into the `model_zoo/team02_faceRes` directory.

Ensure the files are organized with the following structure:

<pre> 
 model_zoo
├── team00_CodeFormer
│   └── team00_CodeFormer.txt
└── team02_faceRes
    ├── FFHQ_eye_mouth_landmarks_512.pth
    ├── GFPGANv1.3.pth
    ├── default
    │   ├── pool_embeds.pt
    │   └── prompt_embeds.pt
    ├── detection_Resnet50_Final.pth
    ├── models--stabilityai--stable-diffusion-3-medium-diffusers
    │   ├── ……
 </pre>

To run our method, execute the following command:

```
python test.py --test_dir <path_to_input_images> --save_dir <path_to_save_results> --model_id 2
```

In the third stage, we apply the ZSSR (Zero-Shot Super-Resolution) approach for image-specific fine-tuning. This step is optional but can be time-consuming. If you would like to generate the restored image using ZSSR, after running the above command, execute the following command:

```
HF_ENDPOINT=https://hf-mirror.com python test.py --test_dir <path_to_input_images> --save_dir <path_to_save_results> --model_id 200

```