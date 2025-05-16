
import importlib

# from models.mvsnet import MVSNet, mvsnet_loss
# from models.rmvsnet import RMVSNet, mvsnet_loss
# from models.casrednet import CascadeREDNet, cas_mvsnet_loss
from models.casrednet1 import CascadeREDNet, cas_mvsnet_loss, Infer_CascadeREDNet

# find the model definition by name, for example casrednet (casrednet.py)
def find_model_def(model_name):
    module_name = 'models.{}'.format(model_name)
    module = importlib.import_module(module_name)
    return getattr(module, "CascadeREDNet"), getattr(module, "cas_mvsnet_loss"), getattr(module, "Infer_CascadeREDNet")
