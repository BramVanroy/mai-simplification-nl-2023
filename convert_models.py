from transformers import TFAutoModelForSeq2SeqLM, FlaxAutoModelForSeq2SeqLM


def main(din: str):
    tf_model = TFAutoModelForSeq2SeqLM.from_pretrained(din, from_pt=True)
    tf_model.save_pretrained(din)
    del tf_model
    flax_model = FlaxAutoModelForSeq2SeqLM.from_pretrained(din, from_pt=True)
    flax_model.save_pretrained(din)
    del flax_model


if __name__ == '__main__':
    import argparse
    cparser = argparse.ArgumentParser(
        description="Convert a given directory with a PyTorch checkpoint to Tensorflow and Flax"
    )
    cparser.add_argument("fin", help="Input directory with torch checkpoint")
    main(cparser.parse_args().fin)
