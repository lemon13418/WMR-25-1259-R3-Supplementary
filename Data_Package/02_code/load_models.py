# -*- coding: utf-8 -*-
"""load_models.py - 加载数据包内模型的兼容加载器 (DNN wrapper 兼容)"""
import os, pickle, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS = os.path.join(ROOT, '03_models')


class SimpleKerasWrapper:
    """与 model_training.py 内 wrapper 同形的反序列化占位 (仅供加载 .pkl 元数据)."""
    def __init__(self, *a, **k):
        self.model = None

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("DNN wrapper requires a loaded .keras model; "
                               "re-run 03_model_training.py (or load the .keras file directly).")
        return self.model.predict(X)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    m = {os.path.basename(p): (p, os.path.getsize(p) // 1024)
         for p in [os.path.join(MODELS, f) for f in os.listdir(MODELS)] if p.endswith('.pkl')}
    print('Available model artifacts (17):')
    for k, (_, kb) in sorted(m.items()):
        print('  %-48s %5d KB' % (k, kb))
    print('\nDNN .pkl files embed a Keras wrapper; to use them, re-run the training '
          'script or load the paired .keras weights with tensorflow.keras.models.load_model.')