import json
import logging
import os
import time
from abc import ABC

import torch
import transformers
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir
                )
            else:
                logger.warning("Missing the operation mode.")

            if self.setup_config["bettertransformer"] == True or os.environ.get("USE_BETTERTRANSFORMER_VAR", "no") == "yes":
                # convert to BetterTransformer
                try:
                    self.model = BetterTransformer.transform(self.model)
                    logger.info(
                        "Successfully transformed the model to use BetterTransformer."
                    )
                except Exception as e:
                    raise Exception(
                        f"Could not convert the model to BetterTransformer, with the error: {e}"
                    )
            else:
                logger.info("Using vanilla PyTorch (not BetterTransformer).")

            self.model.to(self.device)
        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if any(
            fname
            for fname in os.listdir(model_dir)
            if fname.startswith("vocab.") and os.path.isfile(fname)
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.eval()
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning("Missing the index_to_name.json file.")

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """

        input_ids_batch = None
        attention_mask_batch = None

        all_texts = []
        pre_tokenized = True
        for idx, data in enumerate(requests):
            input_data = data.get("data")
            if input_data is None:
                input_data = data.get("body")

            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")

            retrieved_data = json.loads(input_data)

            pre_tokenized = pre_tokenized and retrieved_data["pre_tokenized"]

            if retrieved_data["pre_tokenized"] == False:
                if isinstance(retrieved_data["text"], (bytes, bytearray)):
                    input_text = input_text.decode("utf-8")
                all_texts.append(input_text)

        # logger.info(f"Batched the received text into {all_texts}")
        if pre_tokenized == False:
            logger.info("TOKENIZEING")
            inputs = self.tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            inputs = {}
            inputs["input_ids"] = torch.tensor(retrieved_data["input_ids"])
            inputs["attention_mask"] = torch.tensor(retrieved_data["attention_mask"])

        self.n_pads = (inputs["input_ids"] == 0).sum().item()
        self.n_elems = inputs["input_ids"].numel()
        self.sequence_length = inputs["input_ids"].shape[1]

        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        return (inputs["input_ids"], inputs["attention_mask"])

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []

        # Handling inference for sequence_classification.
        with torch.no_grad():
            predictions = self.model(input_ids_batch, attention_mask_batch)

        logger.info(f"Output size of the text-classification model: {predictions[0].size()}")
        logger.info(f"Output of the text-classification model: {predictions}")

        num_rows, num_cols = predictions[0].shape
        for i in range(num_rows):
            out = predictions[0][i].unsqueeze(0)
            y_hat = out.argmax(1).item()
            predicted_idx = str(y_hat)
            inferences.append(self.mapping[predicted_idx])

        logger.info(f"Processed output: {inferences}")
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        handler_time = round((time.time() - start_time) * 1000, 2)
        peak_gpu_memory = round(torch.cuda.max_memory_allocated("cuda:0") * 10e-6, 2)
        metrics.add_time("HandlerTime", handler_time, None, "ms")
        torch.cuda.reset_peak_memory_stats("cuda:0")

        for i, out in enumerate(output):
            output[i] = (
                out,
                handler_time,
                peak_gpu_memory,
                self.n_pads,
                self.n_elems,
                self.sequence_length,
            )
        return output
