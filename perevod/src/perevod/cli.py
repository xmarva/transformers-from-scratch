import argparse
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="perevod",
        description="Translate English into Russian, one line per argument, or, "
        "without arguments, one line per line of the standard input.",
    )
    parser.add_argument("text", nargs="*", help="sentences to translate")
    parser.add_argument("--beam-size", type=int, help="1 is greedy decoding")
    parser.add_argument("--model", help="a Hugging Face repository or a local directory")
    parser.add_argument("--revision", help="a branch, tag or commit of the repository")
    parser.add_argument("--device", help="cpu or cuda (default: cuda when there is one)")
    parser.add_argument("--version", action="version", version=f"perevod {__version__}")
    args = parser.parse_args(argv)

    texts = args.text
    if not texts:
        texts = [line.rstrip("\n") for line in sys.stdin]

    from .translator import MODEL, Translator  # PyTorch loads only when there is work to do

    translator = Translator.from_pretrained(args.model or MODEL, args.revision, args.device)
    for translation in translator.translate(texts, beam_size=args.beam_size):
        print(translation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
