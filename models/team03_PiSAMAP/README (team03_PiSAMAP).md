# install all requirements
pip install -r requirements.txt

# download model weights
Linkes to all model weights are organized following the organizers' instructions.

# Run
CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 3

# Note

Our method is relatively time-consuming, please preserve at least 2~3 days for inference !
