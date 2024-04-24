from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, make_scheduler
from processor import do_train
from utils.set_seed import set_seed
import torch

seed = 1024
set_seed(seed)

lr = 6e-6
epochs = 20
batch_size = 1
device = "cuda:0"

training_loader, testing_loader = make_dataloader("data/S6070_r.csv", "data/s669.csv", batch_size)
for data_i, data in enumerate(training_loader):
    print(len(data), data[0].shape)
    if data_i == 1:
        break

# --> ESM_ensamble_cls_env, ESM_ensamble_cls_env2, ESM_ensamble_cls_env_attention, ESM_ensamble_cls, ESM_ensamble_env
model = make_model("ESM_ensamble_cls_env_attention")
optimizer_name = "Adam"
optimizer =  make_optimizer(model, lr, optimizer_name=optimizer_name)
scheduler =  make_scheduler(optimizer, lr, training_loader, epochs, scheduler_name="CosineLR")
# scheduler = None

# checkpoint_path = "/mlx/users/mayiming.001/playground/maym/workspace/esm/ddG/prostata_muti_batch_ddp/checkpoint/filt/pretrain_cls_env_attention_cosine_e20.pt"
checkpoint_path = "checkpoint/re_pretrain/pretrain_cls_env_attention_cosine_e4.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
model.to(device)

print(lr, batch_size, seed, optimizer_name, checkpoint_path)

do_train(model, 
         optimizer, 
         scheduler, 
         training_loader, 
         testing_loader, 
         device, 
         epochs)