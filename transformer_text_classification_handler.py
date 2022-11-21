import ast
import json
import logging
import os
from abc import ABC

from optimum.bettertransformer import BetterTransformer
import torch
import transformers

from captum.attr import LayerIntegratedGradients
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
)

from ts.torch_handler.base_handler import BaseHandler

import time

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

        # Loading the shared object of compiled Faster Transformer Library if Faster Transformer is set
        if self.setup_config["FasterTransformer"]:
            faster_transformer_complied_path = os.path.join(
                model_dir, "libpyt_fastertransformer.so"
            )
            torch.classes.load_library(faster_transformer_complied_path)

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir
                )
            elif self.setup_config["mode"] == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"] == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            elif self.setup_config["mode"] == "text_generation":
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            else:
                logger.warning("Missing the operation mode.")

            """
            # convert to BetterTransformer
            try:
                self.model = BetterTransformer.transform(self.model)
                logger.info("Successfully transformed the model to use BetterTransformer.")
            except Exception as e:
                raise Exception(f"Could not convert the model to BetterTransformer, with the error: {e}")
            """

            # HF GPT2 models options can be gpt2, gpt2-medium, gpt2-large, gpt2-xl
            # this basically palce different model blocks on different devices,
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L962
            if (
                self.setup_config["model_parallel"]
                and "gpt2" in self.setup_config["model_name"]
            ):
                self.model.parallelize()
            else:
                self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if "gpt2" in self.setup_config["model_name"]:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", pad_token="<|endoftext|>"
            )

        elif any(
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
        # Question answering does not need the index_to_name.json file.
        if not (
            self.setup_config["mode"] == "question_answering"
            or self.setup_config["mode"] == "text_generation"
        ):
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
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            all_texts.append(input_text)

        logger.info(f"Batched the received text into {all_texts}")


        inputs = self.tokenizer(
                    all_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

        self.n_pads = (inputs["input_ids"] == 0).sum().item()
        self.n_elems = inputs["input_ids"].numel()
        self.sequence_length = inputs["input_ids"].shape[1]

        inputs = inputs.to(self.device)

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
        predictions = self.model(input_ids_batch, attention_mask_batch)
        print(
            "This the output size from the Seq classification model",
            predictions[0].size(),
        )
        print("This the output from the Seq classification model", predictions)

        num_rows, num_cols = predictions[0].shape
        for i in range(num_rows):
            out = predictions[0][i].unsqueeze(0)
            y_hat = out.argmax(1).item()
            predicted_idx = str(y_hat)
            inferences.append(self.mapping[predicted_idx])

        print("Generated text", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(self, input_batch, text, target):
        """This function initialize and calls the layer integrated gradient to get word importance
        of the input text if captum explanation has been selected through setup_config
        Args:
            input_batch (int): Batches of tokens IDs of text
            text (str): The Text specified in the input request
            target (int): The Target can be set to any acceptable label under the user's discretion.
        Returns:
            (list): Returns a list of importances and words.
        """

        if self.setup_config["captum_explanation"]:
            embedding_layer = getattr(self.model, self.setup_config["embedding_name"])
            embeddings = embedding_layer.embeddings
            self.lig = LayerIntegratedGradients(captum_sequence_forward, embeddings)
        else:
            logger.warning("Captum Explanation is not chosen and will not be available")

        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        text_target = ast.literal_eval(text)

        if not self.setup_config["mode"] == "question_answering":
            text = text_target["text"]
        self.target = text_target["target"]

        input_ids, ref_input_ids, attention_mask = construct_input_ref(
            text, self.tokenizer, self.device, self.setup_config["mode"]
        )
        all_tokens = get_word_token(input_ids, self.tokenizer)
        response = {}
        response["words"] = all_tokens
        if (
            self.setup_config["mode"] == "sequence_classification"
            or self.setup_config["mode"] == "token_classification"
        ):

            attributions, delta = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )

            attributions_sum = summarize_attributions(attributions)
            response["importances"] = attributions_sum.tolist()
            response["delta"] = delta[0].tolist()

        elif self.setup_config["mode"] == "question_answering":
            attributions_start, delta_start = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )
            attributions_end, delta_end = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 1, self.model),
                return_convergence_delta=True,
            )
            attributions_sum_start = summarize_attributions(attributions_start)
            attributions_sum_end = summarize_attributions(attributions_end)
            response["importances_answer_start"] = attributions_sum_start.tolist()
            response["importances_answer_end"] = attributions_sum_end.tolist()
            response["delta_start"] = delta_start[0].tolist()
            response["delta_end"] = delta_end[0].tolist()

        return [response]

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
        logger.info("in custom handle!!!!!")
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
        metrics.add_time(
            "HandlerTime", handler_time, None, "ms"
        )
        torch.cuda.reset_peak_memory_stats("cuda:0")

        for i, out in enumerate(output):
            output[i] = (out, handler_time, peak_gpu_memory, self.n_pads, self.n_elems, self.sequence_length)
        return output



def construct_input_ref(text, tokenizer, device, mode):
    """For a given text, this function creates token id, reference id and
    attention mask based on encode which is faster for captum insights
    Args:
        text (str): The text specified in the input request
        tokenizer (AutoTokenizer Class Object): To word tokenize the input text
        device (cpu or gpu): Type of the Environment the server runs on.
    Returns:
        input_id(Tensor): It attributes to the tensor of the input tokenized words
        ref_input_ids(Tensor): Ref Input IDs are used as baseline for the attributions
        attention mask() :  The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them.
    """
    if mode == "question_answering":
        question_context = ast.literal_eval(text)
        question = question_context["question"]
        context = question_context["context"]
        text_ids = tokenizer.encode(question, context, add_special_tokens=False)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    logger.info("text_ids %s", text_ids)
    logger.info("[tokenizer.cls_token_id] %s", [tokenizer.cls_token_id])
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    logger.info("input_ids %s", input_ids)

    input_ids = torch.tensor([input_ids], device=device)
    # construct reference token ids
    ref_input_ids = (
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * len(text_ids)
        + [tokenizer.sep_token_id]
    )
    ref_input_ids = torch.tensor([ref_input_ids], device=device)
    # construct attention mask
    attention_mask = torch.ones_like(input_ids)
    return input_ids, ref_input_ids, attention_mask


def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    """This function is used to get the predictions from the model and this function
    can be used independent of the type of the BERT Task.
    Args:
        inputs (list): Input for Predictions
        attention_mask (list, optional): The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them, it defaults to None.
        position (int, optional): Position depends on the BERT Task.
        model ([type], optional): Name of the model, it defaults to None.
    Returns:
        list: Prediction Outcome
    """
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred


def summarize_attributions(attributions):
    """Summarises the attribution across multiple runs
    Args:
        attributions ([list): attributions from the Layer Integrated Gradients
    Returns:
        list : Returns the attributions after normalizing them.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_word_token(input_ids, tokenizer):
    """constructs word tokens from token id using the BERT's
    Auto Tokenizer
    Args:
        input_ids (list): Input IDs from construct_input_ref method
        tokenizer (class): The Auto Tokenizer Pre-Trained model object
    Returns:
        (list): Returns the word tokens
    """
    indices = input_ids[0].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)
    # Remove unicode space character from BPE Tokeniser
    tokens = [token.replace("Ġ", "") for token in tokens]
    return tokens
