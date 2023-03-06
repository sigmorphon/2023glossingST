import torch
from transformers import RobertaForTokenClassification, RobertaConfig, TrainingArguments, Trainer
import click
import numpy as np
import wandb
from data import prepare_dataset, load_data_file, create_encoder, write_predictions, ModelType
from custom_tokenizers import tokenizers
from encoder import MultiVocabularyEncoder, special_chars, load_encoder
from eval import eval_morpheme_glosses, eval_word_glosses
from datasets import DatasetDict
from typing import Optional

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_model(encoder: MultiVocabularyEncoder, sequence_length):
    print("Creating model...")
    config = RobertaConfig(
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[2]) + len(special_chars)
    )
    model = RobertaForTokenClassification(config)
    print(model.config)
    return model.to(device)


def create_trainer(model: RobertaForTokenClassification, dataset: Optional[DatasetDict], encoder: MultiVocabularyEncoder, batch_size, lr, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        print(preds)
        decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=2)
        print(decoded_preds[0:1])

        # Decode (gold) labels
        print(labels)
        labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=2)
        print(decoded_labels[0:1])

        if encoder.segmented:
            return eval_morpheme_glosses(pred_morphemes=decoded_preds, gold_morphemes=decoded_labels)
        else:
            return eval_word_glosses(pred_words=decoded_preds, gold_words=decoded_labels)

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["dev"] if dataset else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    return trainer


languages = {
    'arp': 'Arapaho',
    'git': 'Gitksan',
    'lez': 'Lezgi',
    'nyb': 'Nyangbo',
    'ddo': 'Tsez',
    'usp': 'Uspanteko'
}


@click.command()
@click.argument('mode')
@click.option("--lang", help="Which language to train", type=str, required=True)
@click.option("--track", help="[closed, open] whether to use morpheme segmentation", type=str, required=True)
@click.option("--pretrained_path", help="Path to pretrained model", type=click.Path(exists=True))
@click.option("--encoder_path", help="Path to pretrained encoder", type=click.Path(exists=True))
@click.option("--data_path", help="The dataset to run predictions on. Only valid in predict mode.", type=click.Path(exists=True))
def main(mode: str, lang: str, track: str, pretrained_path: str, encoder_path: str, data_path: str):
    if mode == 'train':
        wandb.init(project="igt-generation", entity="michael-ginn")

    MODEL_INPUT_LENGTH = 512

    is_open_track = track == 'open'
    print("IS OPEN", is_open_track)

    train_data = load_data_file(f"../../data/{languages[lang]}/{lang}-train-track{'2' if is_open_track else '1'}-uncovered")
    dev_data = load_data_file(f"../../data/{languages[lang]}/{lang}-dev-track{'2' if is_open_track else '1'}-uncovered")

    print("Preparing datasets...")

    tokenizer = tokenizers['morpheme_no_punc' if is_open_track else 'word_no_punc']

    if mode == 'train':
        encoder = create_encoder(train_data, tokenizer=tokenizer, threshold=1,
                                 model_type=ModelType.TOKEN_CLASS, split_morphemes=is_open_track)
        encoder.save()
        dataset = DatasetDict()
        dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizer, encoder=encoder,
                                           model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizer, encoder=encoder,
                                         model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(model, dataset=dataset, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=80)

        print("Training...")
        trainer.train()
        print("Saving model to ./output")
        trainer.save_model('./output')
        print("Model saved at ./output")
    elif mode == 'predict':
        encoder = load_encoder(encoder_path)
        if not hasattr(encoder, 'segmented'):
            encoder.segmented = is_open_track
        print("ENCODER SEGMENTING INPUT: ", encoder.segmented)
        predict_data = load_data_file(data_path)
        predict_data = prepare_dataset(data=predict_data, tokenizer=tokenizer, encoder=encoder,
                                       model_input_length=MODEL_INPUT_LENGTH, model_type=ModelType.TOKEN_CLASS, device=device)
        model = RobertaForTokenClassification.from_pretrained(pretrained_path)
        trainer = create_trainer(model, dataset=None, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=50)
        preds = trainer.predict(test_dataset=predict_data).predictions
        write_predictions(data_path, lang=lang, preds=preds, pred_input_data=predict_data, encoder=encoder, from_vocabulary_index=2)


if __name__ == "__main__":
    main()
