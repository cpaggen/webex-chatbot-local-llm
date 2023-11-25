# WebEx Chatbot with local LLM inference

Minimal and sufficient codebase to run a WebEx chatbot that performs local inference from any model you fancy.
The benefit is that this does not require any OpenAI API key, it's entirely free - as in "free lunch".
The code runs without a GPU, but if patience is not your forte then I heavily recommend at least one powerful GPU.

## Installation

First, download a model such as [SynthIA-7B-v2.0](https://huggingface.co/TheBloke/SynthIA-7B-v2.0-16k-GGUF/tree/main)
Ensure you have created a WebEx bot (see [instructions](https://developer.webex.com/docs/bots) and make sure you have a valid WebEx API access token. You'll need to export that token as an environment variable.

### With GPU offload

If you want GPU offload, install the device drivers and the CUDA toolkit (assuming a CUDA-compatible GPU).
Verify that `nvcc` is in your path and executes correctly.
This code uses the Llama-cpp library which supports CPU and GPU inference. Turn on verbose mode in the code to ensure your are indeed offloading to the GPU. Install llama-cpp as follows for GPU support:

```
CUDACXX="/usr/local/cuda-12.3/bin/nvcc" \
  CMAKE_CUDA_COMPILER="path to nvcc"  \    
  CMAKE_ARGS="-DLLAMA_CUBLAS=on"      \    
  FORCE_CMAKE=1 pip install --upgrade \    
  --force-reinstall llama-cpp-python --no-cache-dir
```
Then run `pip install -r requirements.txt`

### Without GPU offload

Just run `pip install -r requirements.txt` and everything should just work! 
It runs quite slowly without a GPU, so be patient after asking a question. Wait times beyond a minute are normal.

## Sample output with GPU offload

```
chris@c245-a16-1:~/LOCALBOT$ python localBot.py
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 4 CUDA devices:
  Device 0: NVIDIA A16, compute capability 8.6
  Device 1: NVIDIA A16, compute capability 8.6
  Device 2: NVIDIA A16, compute capability 8.6
  Device 3: NVIDIA A16, compute capability 8.6
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from /home/chris/MODELS/synthia-7b-v2.0-16k.Q4_K_M.gguf (version GGUF V3 (latest))
...
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = mostly Q4_K - Medium
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 4.07 GiB (4.83 BPW)
llm_load_print_meta: general.name   = nurtureai_synthia-7b-v2.0-16k
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: PAD token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MiB
llm_load_tensors: using CUDA for GPU acceleration
ggml_cuda_set_main_device: using device 0 (NVIDIA A16) as main device
llm_load_tensors: mem required  =  172.97 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloaded 32/35 layers to GPU
llm_load_tensors: VRAM used: 3992.50 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: kv self size  =  512.00 MiB
llama_build_graph: non-view tensors processed: 740/740
llama_new_context_with_model: compute buffer total size = 579.07 MiB
llama_new_context_with_model: VRAM scratch buffer: 576.01 MiB
llama_new_context_with_model: total VRAM used: 4568.51 MiB (model: 3992.50 MiB, context: 576.01 MiB)
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 |
2023-11-25 10:16:31  [INFO]  [localBot.__main__.main]:67 LLM loaded localy
LLM loaded localy
2023-11-25 10:16:31  [INFO]  [webex_bot.webex_bot.webex_bot.__init__]:49 Registering bot with Webex cloud
Registering bot with Webex cloud
2023-11-25 10:16:31  [INFO]  [restsession.webexteamssdk.restsession.user_agent]:169 User-Agent: webexteamssdk/0+unknown {"implementation": {"name": "CPython", "version": "3.10.12"}, "system": {"name": "Linux", "release": "5.15.0-89-generic"}, "cpu": "x86_64", "organization": {}}
User-Agent: webexteamssdk/0+unknown {"implementation": {"name": "CPython", "version": "3.10.12"}, "system": {"name": "Linux", "release": "5.15.0-89-generic"}, "cpu": "x86_64", "organization": {}}
2023-11-25 10:16:32  [INFO]  [webex_bot.webex_bot.webex_bot.get_me_info]:90 Running as bot 'aci-bot-cpaggen' with email ['aci-bot-cpaggen@webex.bot']
Running as bot 'aci-bot-cpaggen' with email ['aci-bot-cpaggen@webex.bot']
2023-11-25 10:16:32  [INFO]  [localBot.__main__.main]:73 WebEx bot init done
WebEx bot init done
2023-11-25 10:16:33  [INFO]  [webex_websocket_client.webex_bot.websockets.webex_websocket_client._connect_and_listen]:160 Opening websocket connection to wss://mercury-connection-partition0-a.wbx2.com/v1/apps/wx2/registrations/a6c45226-0000-0000-0000-544349d23cbf/messages
Opening websocket connection to wss://mercury-connection-partition0-a.wbx2.com/v1/apps/wx2/registrations/a6c45226-0000-0000-0000-544349d23cbf/messages
2023-11-25 10:16:33  [INFO]  [webex_websocket_client.webex_bot.websockets.webex_websocket_client._connect_and_listen]:163 WebSocket Opened.
WebSocket Opened.
2023-11-25 10:18:37  [INFO]  [INFO]  [localBot.__main__.execute]:45 Querying LLM question=what type of music does Rush play?
Querying LLM question=what type of music does Rush play?
Llama.generate: prefix-match hit

llama_print_timings:        load time =     149.59 ms
llama_print_timings:      sample time =      14.79 ms /    87 runs   (    0.17 ms per token,  5881.16 tokens per second)
llama_print_timings: prompt eval time =     115.72 ms /     9 tokens (   12.86 ms per token,    77.77 tokens per second)
llama_print_timings:        eval time =    4365.71 ms /    86 runs   (   50.76 ms per token,    19.70 tokens per second)
llama_print_timings:       total time =    4679.80 ms
2023-11-25 10:20:40  [INFO]  [webex_websocket_client.webex_bot.websockets.webex_websocket_client._ack_message]:108 WebSocket ack message with id=Y2lzY29zcGFyazovL3VzL01FU1NBR0UvMTYyNTI1MTAtOGI3ZC0xMWVlLThiMzMtNDUxY2YwZDMwYzkz. Complete.
WebSocket ack message with id=Y2lzY29zcGFyazovL3VzL01FU1NBR0UvMTYyNTI1MTAtOGI3ZC0xMWVlLThiMzMtNDUxY2YwZDMwYzkz. Complete.
```

The response visible to the user inside the WebEx Teams app is:

```
Rush is a Canadian rock band that was formed in 1968. The band's music has been described as progressive rock, hard rock, and heavy metal. They are known for their complex compositions, virtuosic musicianship, and concept albums. Some of their most popular songs include "Tom Sawyer," "The Spirit of Radio," "Limelight," and "Freewill."
```

