import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFERENCES = {
    "CVPR": "computer vision, image processing, visual recognition, object detection, segmentation, 3D, pose estimation, GANs, SLAM, video analysis.",
    "TMLR": "theoretical machine learning, generalization bounds (non-neural), probabilistic models, Bayesian inference, causal learning, meta-learning, diffusion models, Wasserstein distance.",
    "EMNLP": "natural language processing, linguistic analysis, machine translation, language generation, crowdsourcing, sentiment analysis, NER, summarization, discourse, text embeddings, transformer models.",
    "NeurIPS": "deep learning, neural networks, ReLU, generalization bounds (neural networks), reinforcement learning,  multi-agent systems, generative models, causal inference, interdisciplinary AI, large-scale learning, optimization, scalability.",
    "KDD": "recommendation systems, data mining, big data, graph mining, anomaly detection, time-series, federated learning, IoT data, cloud computing."
}
