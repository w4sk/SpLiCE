import torch
from .model import SPLICE
import os
import urllib

GITHUB_HOST_LINK = "https://raw.githubusercontent.com/AI4LIFE-GROUP/SpLiCE/main/data/"

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

SUPPORTED_MODELS = {
    "clip": [
        "ViT-B/32",
        "ViT-B/16",
        "RN50"
    ],
    "open_clip": [
        "ViT-B-32"
    ]
}

SUPPORTED_VOCAB = [
    "laion",
    "laion_bigrams",
    "mscoco"
]

def available_models():
    """Returns supported models."""
    return SUPPORTED_MODELS

def _download(url: str, root: str, subfolder: str):
    """_download

    Parameters
    ----------
    url : str
        Link to download files from
    root : str
        Destination folder
    subfolder : str
        Subfolder (either /vocab or /means)

    Returns
    -------
    str
        A path to the desired file
    """
    root_subfolder = os.path.join(root, subfolder)
    os.makedirs(root_subfolder, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root_subfolder, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    return download_target

def load(name: str, vocabulary: str, vocabulary_size: int = -1, device = None, download_root = None, **kwargs):
    """load SpLiCE

    Parameters
    ----------
    name : str
        the name of the CLIP backbone used
    vocabulary : str
        the vocabulary set used for dictionary learning
    device : Union[str, torch.device], optional
        torch device
    download_root : str
        path to download vocabulary and mean data to, otherwise "~/.cache/splice"
    """
    # Set device if not specified
    if device is None:
        device = get_device()
    
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                clip_backbone, _ = clip.load(model_name, device=device)
                tokenizer = clip.tokenize
            elif library == "open_clip":
                import open_clip
                # Use create_model_and_transforms to properly load pretrained weights
                # This handles Hugging Face downloading correctly
                clip_backbone, _, _ = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained='laion2b_s34b_b79k'
                )
                clip_backbone = clip_backbone.to(device)
                tokenizer = open_clip.get_tokenizer(model_name)
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
    
    # Check for custom local vocabulary first
    # __file__ is in SpLiCE/splice/, so go up one level to SpLiCE/
    local_vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vocab", vocabulary + ".txt")
    
    if os.path.exists(local_vocab_path):
        # Load custom local vocabulary
        concepts = []
        vocab = []
        vocab_path = local_vocab_path

        concept_root = download_root or os.path.expanduser("~/.cache/splice/")
        os.makedirs(os.path.join(concept_root, "embeddings"), exist_ok=True)

        if vocabulary_size <= 0:
            vocabulary_size_name = "full"
        else:
            vocabulary_size_name = vocabulary_size
        concept_path = os.path.join(concept_root, f"embeddings/{name}_{vocabulary}_{vocabulary_size_name}_embeddings.pt")

        if os.path.isfile(concept_path):
            concepts = torch.load(concept_path, map_location=torch.device(device))
        else:
            with open(vocab_path, "r") as f:
                lines = f.readlines()
                if vocabulary_size > 0 and vocabulary_size < len(lines):
                    lines = lines[:vocabulary_size]  # Use first N for custom vocab
                for line in lines:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    vocab.append(line)
                    tokens = tokenizer(line).to(device)
                    with torch.no_grad():
                        concept_embedding = clip_backbone.encode_text(tokens)
                    concepts.append(concept_embedding)
            
            concepts = torch.nn.functional.normalize(torch.stack(concepts).squeeze(), dim=1)
            concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)
            torch.save(concepts, concept_path)
    elif vocabulary in SUPPORTED_VOCAB:
        concepts = []
        vocab = []

        vocab_path = _download(os.path.join(GITHUB_HOST_LINK, "vocab", vocabulary + ".txt"), download_root or os.path.expanduser("~/.cache/splice/"), "vocab")

        concept_root = download_root or os.path.expanduser("~/.cache/splice/")
        os.makedirs(os.path.join(concept_root, "embeddings"), exist_ok=True)

        if vocabulary_size <= 0:
            vocabulary_size_name = "full"
        else:
            vocabulary_size_name = vocabulary_size
        concept_path = os.path.join(concept_root, f"embeddings/{name}_{vocabulary}_{vocabulary_size_name}_embeddings.pt")

        if os.path.isfile(concept_path):
            concepts = torch.load(concept_path, map_location=torch.device(device))
        else:
            with open(vocab_path, "r") as f:
                lines = f.readlines()
                if vocabulary_size > 0:
                    lines = lines[-vocabulary_size:]
                for line in lines:
                    line = line.strip()
                    vocab.append(line)
                    tokens = tokenizer(line).to(device)
                    with torch.no_grad():
                        concept_embedding = clip_backbone.encode_text(tokens)
                    concepts.append(concept_embedding)
            
            concepts = torch.nn.functional.normalize(torch.stack(concepts).squeeze(), dim=1)
            concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)
            torch.save(concepts, concept_path)
    else:
        raise RuntimeError(f"Vocabulary {vocabulary} not supported and not found in local data/vocab/ directory.")
    
    
    model_path = model_name.replace("/","-")
    mean_path = _download(os.path.join(GITHUB_HOST_LINK, "means", f"{library}_{model_path}_image.pt"), download_root or os.path.expanduser("~/.cache/splice/"), "means")
    image_mean = torch.load(mean_path, map_location=torch.device(device))
    splice = SPLICE(
        image_mean=image_mean,
        dictionary=concepts,
        clip=clip_backbone,
        device=device,
        **kwargs
    )

    return splice

def get_vocabulary(name: str, vocabulary_size: int, download_root = None):
    """get_vocabulary: Gets a list of vocabulary for use in mapping sparse weight vectors to text.

    Parameters
    ----------
    name : str
        Supported vocabulary type. Either 'mscoco' or 'laion'.
    vocabulary_size : int
        Number of concepts to consider. Will consider highest frequency concepts.
    download_root : str, optional
        If specified, where to access vocab txt file from, otherwise will use default "~/.cache/splice/vocab", by default None

    Returns
    -------
    _type_
        _description_
    """
    # Check for custom local vocabulary first
    # __file__ is in SpLiCE/splice/, so go up one level to SpLiCE/
    local_vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vocab", name + ".txt")
    
    if os.path.exists(local_vocab_path):
        vocab_path = local_vocab_path
    elif name in SUPPORTED_VOCAB:
        vocab_path = _download(os.path.join(GITHUB_HOST_LINK, "vocab", name + ".txt"), download_root or os.path.expanduser("~/.cache/splice/"), "vocab")
    else:
        raise RuntimeError(f"Vocabulary {name} not supported and not found in local data/vocab/ directory.")

    vocab = []
    with open(vocab_path, "r") as f:
        lines = f.readlines()
        if vocabulary_size > 0 and vocabulary_size < len(lines):
            # For custom vocab, use first N; for supported vocab, use last N (frequency-based)
            if os.path.exists(local_vocab_path):
                lines = lines[:vocabulary_size]
            else:
                lines = lines[-vocabulary_size:]
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:  # Skip empty lines
                vocab.append(line_stripped)
    return vocab

def get_tokenizer(name: str):
    """get_tokenizer Gets tokenizer for SpLiCE model

    Parameters
    ----------
    name : str
        SpLiCE model

    Returns
    -------
    _type_
        CLIP backbone tokenizer
    """
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                return clip.tokenize
            elif library == "open_clip":
                import open_clip
                return open_clip.get_tokenizer(model_name)
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
def get_preprocess(name: str):
    """get_preprocess Gets image preprocessing transform

    Parameters
    ----------
    name : str
        SpLiCE model

    Returns
    -------
    _type_
        CLIP backbone preprocessing transform.
    """
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                return clip.load(model_name)[1]
            elif library == "open_clip":
                import open_clip
                return open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')[2]
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
def decompose_dataset(dataloader, splicemodel=None, device="cpu"):
    """decompose_dataset decomposes a full dataset and returns the mean weights of the sparse decomposition.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        Dataloader that returns (image, label) tuples for decomposition. 
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device.
    Returns
    -------
    weights : torch.tensor
        A vector of the mean value of sparse weights over the dataset.
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=-1, l1_penalty=0.15, return_weights=True,device=device)
    splicemodel.eval()

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    ret_weights = None
    l0 = 0
    cosine = 0
    total = 0

    for data in dataloader:
        try: ## Handle dataloaders of just images or images and labels
            image, _ = data
        except:
            image = data
        image = image.to(device)

        with torch.no_grad():

            (batch_weights, batch_cosine) = splicemodel.encode_image(image)
            if ret_weights is None:
                ret_weights = torch.sum(batch_weights, dim=0)
            else:
                ret_weights += torch.sum(batch_weights, dim=0)
            
            l0 += torch.linalg.vector_norm(batch_weights, dim=1, ord=0).sum().item()
            cosine += batch_cosine.item()
            total += image.shape[0]
        
    return ret_weights/total, l0/total, cosine/total

def decompose_classes(dataloader, target_label, splicemodel=None, device="cpu"):
    """decompose_dataset decomposes a full dataset and returns the mean weights of the sparse decomposition per class.

    Parameters
    ----------
    dataloader : torch Dataloader
        Dataloader that returns (image, label) tuples for decomposition
    target_label : int optional
        Specific class label to decompose
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device
    

    Returns
    -------
    class_weights : dict
        A dictionary of elements {label : mean sparse weight vector}
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=-1, l1_penalty=0.15, return_weights=True, return_cosine=True, device=device)
    splicemodel.eval()

    class_weights={}
    class_totals={}

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    l0 = 0
    cosine = 0
    total = 0

    for idx, (image, label) in enumerate(dataloader):

        if target_label != None:
            idx = torch.argwhere(label == target_label).squeeze()
            if idx.nelement() == 0:
                continue

            image = image[idx]
            label = label[idx]

            if idx.nelement() == 1:
                image, label = image.unsqueeze(0), label.unsqueeze(0)
            
            

        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)

            (weights, batch_cosine) = splicemodel.encode_image(image)

            for i in range(image.shape[0]):
                weights_i, label_i = weights[i], label[i].item()
                if label_i in class_weights:
                    class_weights[label_i] += weights_i
                    class_totals[label_i] += 1
                else:
                    class_weights[label_i] = weights_i
                    class_totals[label_i] = 1

            l0 += torch.linalg.vector_norm(weights, dim=1, ord=0).sum().item()
            cosine += batch_cosine.item()
            total += image.shape[0]
    
    for label in class_weights.keys():
        class_weights[label] /= class_totals[label]

    return class_weights, l0/total, cosine/total
    

def decompose_image(image, splicemodel=None, device="cpu"):
    """decompose_image _summary_

    Parameters
    ----------
    image : torch tensor
        A preprocessed image to decompose
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device.
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=-1, l1_penalty=0.15, return_weights=True, device=device)
    splicemodel.eval()

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    (weights, cosine) = splicemodel.encode_image(image.to(device))
    l0_norm = torch.linalg.vector_norm(weights.squeeze(), ord=0).item()

    return weights, l0_norm, cosine.item()
