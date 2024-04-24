import torch
from datasets import make_dataloader
from model import make_model
from processor import do_test
from utils.set_seed import set_seed

batch_size = 8
device = "cuda:0"

set_seed(1024)
# --> ssym, s669, p53, myoglobin, CAGI5_TPMT_PTEN
testing_loader = make_dataloader(test_dataset_path = "data/S669_reverse_processed.csv", batch_size=batch_size, is_training=False)
# --> ESM_ensamble_cls, ESM_ensamble_pos, ESM_ensamble_env, ESM_ensamble_cls_pos, ESM_ensamble_cls_env, ESM_ensamble_pos_env, ESM_ensamble_cls_env_attention
model = make_model("ESM_ensamble_cls_env_attention")
checkpoint = torch.load("checkpoint/task13/myoglobin.pt", map_location="cpu")
model.load_state_dict(checkpoint)

do_test(
    model,
    testing_loader,
    device
)