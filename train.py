import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import copy
from omegaconf import DictConfig
# 設定資料集與NeMo Model位置
nemoModelPath = './'
datasetPath = './dataset'

quartznet = nemo_asr.models.EncDecCTCModel.restore_from(
    nemoModelPath + '/stt_zh_quartznet15x5.nemo')

# 讀取NeMo Pipeline Config
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = datasetPath + '/config.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
# 設定train data與validation data
params['model']["train_ds"]['manifest_filepath'] = datasetPath + '/train.json'
params['model']["validation_ds"]['manifest_filepath'] = datasetPath + \
    '/validation.json'

# 設定training需要的gpu與多少epochs
trainer = pl.Trainer(gpus=[1], max_epochs=100)

# 設定learning rate
new_opt = copy.deepcopy(params['model']['optim'])
new_opt['lr'] = 0.001

# 設定label、optimization、training data、validation data
quartznet.change_vocabulary(
    new_vocabulary=params['labels']
)
quartznet.setup_optimization(optim_config=DictConfig(new_opt))
quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
quartznet.setup_validation_data(
    val_data_config=params['model']['validation_ds'])

# 訓練model與儲存model
trainer.fit(quartznet)

quartznet.save_to(nemoModelPath + '/first_model.nemo')
