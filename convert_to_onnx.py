from numbers import Number
from typing import List, Optional
from demucs.apply import apply_model
from demucs.htdemucs import HTDemucs
from demucs.hdemucs import HDemucs
import torch.onnx
from torch import Tensor, compile
import math
from demucs.hdemucs import pad1d
from demucs.spec import spectro
from demucs import pretrained
from demucs.states import load_model

audio_len = 44100
mix = torch.randn(1, 2, audio_len * 10) # simulating 10 seconds of audio

#model = pretrained.get_model('htdemucs') # to grab the model url
url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th" # HTDemucs

th = torch.hub.load_state_dict_from_url(
            url, check_hash=True) # type: ignore

#th['klass'] = HDemucs
assert th['klass'] == HTDemucs

th['kwargs']['use_train_segment'] = False

model = load_model(th)

model.eval()

torch.onnx.export(model, 
                (mix),
                "htdemucs.onnx", 
                export_params=True,
                opset_version=20,
                dynamo=True,
                verify=True,
                report=True,
            ).save("htdemucs.onnx")