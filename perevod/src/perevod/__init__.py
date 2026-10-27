__version__ = "0.2.0"


def __getattr__(name):
    # `import perevod` stays light, so that --help and --version answer at once;
    # PyTorch is loaded only when someone asks for the translator
    if name == "Translator":
        from .translator import Translator

        return Translator
    raise AttributeError(f"module 'perevod' has no attribute {name!r}")
