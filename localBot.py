################################################
# Bare minimum code to perform local inference #
# on a pre-trained model. This code runs with  #
# or without GPUs. For GPU offload, you will   #
# need to ensure you have install the CUDA     #
# toolkit and that nvcc is in your path. Then  #
# force pip to make llama-cpp like so          #
# CUDACXX="/usr/local/cuda-12.3/bin/nvcc" \    #
#     CMAKE_CUDA_COMPILER="path to nvcc"  \    #
#     CMAKE_ARGS="-DLLAMA_CUBLAS=on"      \    #
#     FORCE_CMAKE=1 pip install --upgrade \    #
#     --force-reinstall llama-cpp-python       #
#     --no-cache-dir                           #
#                                              #
# Uses the Synthia-7b-Q4_K_M model; feel free  #
# to change and try (visit HuggingFace)        #
#                                              #
# cpaggen Nov 2023 - v1.0 - PoC code           #
################################################

import os
import sys
import logging
from langchain_community.llms import LlamaCpp
from   webex_bot.webex_bot import WebexBot
from   webex_bot.models.command import Command

# enable verbose to debug the LLM's operation
verbose = True

# configure logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log = logging.getLogger(__name__)

class LLM_command(Command):
    def __init__(self, llm):
        super().__init__(
            command_keyword="hello",
            help_message="Just ask me a question",
            card=None)
        self.llm = llm

    def execute(self, message, attachment_actions, activity):
        log.info(f"Querying LLM question={message}")
        resp = self.llm(message,
                        max_tokens=4096,
                        temperature=0.2,
                        # nucleus sampling (mass probability index)
                        top_p=0.1
                        )
        return resp

def main():
    # load the model locally
    llm = LlamaCpp(
        model_path="synthia-7b.gguf",
        n_ctx=4096,
        # number of layers to offload to the GPU
        # do not set to maximum (35 layers for this model)
        n_gpu_layers=32,
        n_batch=1024,
        # half-precision for FP16 must be set to 1 per langchain
        f16_kv=True,  
        verbose=verbose,
        )
    log.info("LLM loaded localy")
    # init WebEx bot handler
    bot = WebexBot(teams_bot_token=access_token,
                   approved_domains=["cisco.com"],
                   bot_name="aci-bot-cpaggen",
                   )
    log.info("WebEx bot init done")
    bot.commands.clear()
    # bot is not programmed with any commands
    # it defaults to the help context, passing the user's question
    bot.help_command = LLM_command(llm)
    bot.run()
    log.info("WebEx bot started")

if __name__ == "__main__":
    access_token    = os.getenv('WEBEX_TEAMS_ACCESS_TOKEN')
    if access_token:
        main()
    else:
        sys.exit("I need a WebEx bot token!")
