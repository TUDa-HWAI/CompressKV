from importlib.metadata import version
import warnings
import transformers
from methods.pyramidkv.llama_hijack import (
    llama_flash_attn2_forward,
    llama_model_forward
    )
from methods.pyramidkv.mistral_hijack import (
    mistral_flash_attn2_forward,
    mistral_model_forward
    )
from methods.pyramidkv.qwen2_hijack import (
    qwen2_flash_attn2_forward,
    qwen2_model_forward
    )
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama():
    # transformers_version = check_version()
    # version_list = ['4.43']
    # warning_flag = True
    # for version in version_list:
    #     if version in transformers_version:
    #         warning_flag = False
    #         break
    # if warning_flag:
    #     warnings.warn(
    #         f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward


def replace_mistral():
    # transformers_version = check_version()
    # version_list = ['4.43']
    # warning_flag = True
    # for version in version_list:
    #     if version in transformers_version:
    #         warning_flag = False
    #         break
    # if warning_flag:
    #     warnings.warn(
    #         f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward

def replace_qwen2():
    # transformers_version = check_version()
    # version_list = ['4.43']
    # warning_flag = True
    # for version in version_list:
    #     if version in transformers_version:
    #         warning_flag = False
    #         break
    # if warning_flag:
    #     warnings.warn(
    #         f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward