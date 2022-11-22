# BetterTransformer Demo

This repo is the source code behind the [Gradio Space](https://huggingface.co/spaces/fxmarty/bettertransformer-demo) demo of BetterTransformer integration with 🤗 Transformers, using [🤗 Optimum](https://github.com/huggingface/optimum/) library.

Built on either [TorchServe](https://github.com/pytorch/serve) or [HF's Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index), this repo helps to understand in a visual and interactive way where BetterTransformer can be useful in production.

By default, this demo uses `distilbert-base-uncased-finetuned-sst-2-english` with a maximum batch size of 8.

## Run on an AWS instance

The example is run on an AWS EC2 g4dn instance, which uses a T4 NVIDIA GPU. Using a basic ubuntu instance, run


```
sudo apt update && sudo apt upgrade
```

Then, we recommend installing docker following the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) guide, and installing the NVIDIA drivers through the package manager and nvidia drivers (for example `nvidia-driver-520`). You may need to add yourself in the `docker` group (`usermod -aG docker ubuntu`). You may need to reboot after the install.

Finally, download the Dockerfile needed to run the TorchServe API:

`wget https://github.com/fxmarty/bettertransformer_demo/raw/main/Dockerfile`

Build the docker (for BetterTransformer, use `ts_config_bettertransformer.properties` and `distilbert_sst2_bettertransformer` args and `--build-arg USE_BETTERTRANSFORMER=yes`):

```
docker build -f Dockerfile \
--build-arg PROP_PATH=./ts_config_vanilla.properties \
--build-arg MAR_NAME=distilbert_sst2_vanilla \
-t bettertransformer-demo .
```

Run the TorchServe server:

```
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 bettertransformer-demo:latest
```

To run in background, use `nohup` without the `-it` argument.

## Check the server is successful running

Use `curl http://127.0.0.1:8080/ping` to check the TorchServe server is running well.

You can as well try:

```python
# outdated example, will not work
import requests

headers = {"Content-Type": "text/plain"}
address = "http://127.0.0.1:8080/predictions/my_tc"  # change this IP if needed
data = "this is positive lol"

response = requests.post(address, headers=headers, data=data)

print(response.status_code)
print(response.text)
```

## Run in a Space

An a demo for BetterTransformer is available at: https://huggingface.co/spaces/fxmarty/bettertransformer-demo . I'll host the demo for a week with two AWS EC2 instances, but since hosting the Space is ~1$/hour, please host one yourself in the future if you want to reproduce.

In any case, example ouputs are available in the Space, to get an idea of the gains of BetterTransformer for latency/throughput!
