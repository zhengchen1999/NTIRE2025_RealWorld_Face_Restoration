import pdb
def pipe(model_dir, input_path=None, output_path=None, device=None, args=None):
    from models.team10_cfDiffbir.CodeFormer import main as CodeFormer
    import os
    # pdb.set_trace()
    CodeFormer(model_dir=model_dir+"/codeformer_weights", input_path=input_path, output_path=output_path, device=device)

    # pdb.set_trace()
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DiffBIR/weights')
    if os.path.exists(target):
        print(f"{target} already exists. Removing it...")
        os.remove(target)
    os.symlink(os.path.abspath(model_dir+"/diffbir_weights"), target)
    print(f"Create a symbolic link from {os.path.abspath(model_dir+'/diffbir_weights')} to {target}. ")
    #os.system(f"cd {os.path.dirname(os.path.abspath(__file__))}/DiffBIR")
    os.system(f'python {os.path.dirname(os.path.abspath(__file__))}/DiffBIR/inference.py --input {os.path.abspath(output_path)} --output {os.path.abspath(output_path)} --cfg_scale 4 --pos_prompt "Good image, Sharp image, sharp edges, High resolution image, Noise-free image" --neg_prompt "bad image, blurry image, blurry edges, low resolution image, noisy image"')
    #os.system(f"cd {os.path.dirname(os.path.abspath(__file__))}/DiffBIR")