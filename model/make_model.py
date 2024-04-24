from .models import ESM_ensamble_cls, ESM_ensamble_pos, ESM_ensamble_env
from .models import ESM_ensamble_cls_pos, ESM_ensamble_cls_env, ESM_ensamble_pos_env, ESM_ensamble_cls_env_attention, ESM_ensamble_cls_env2

def make_model(model_name):
    model_class = globals()[model_name]
    model = model_class()
    return model